import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
import pdb


class Adapter(nn.Module):
    def __init__(self, dim=512, rank=8):
        super(Adapter, self).__init__()
        self.adapter_down = nn.Linear(dim, rank)
        self.adapter_up = nn.Linear(rank, dim)
        self.adapter_mid = nn.Linear(rank, rank)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.drop(x_down)
        x_up = self.adapter_up(x_down)
        return x_up


class MoBA(nn.Module):
    def __init__(self, num_experts=8, dim=512):
        super(MoBA, self).__init__()
        self.num_experts = num_experts
        self.drop = nn.Dropout(0.1)
        self.experts = nn.ModuleList([Adapter(dim, rank=8) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        gating_scores = self.gate(x)
        gating_weights = F.softmax(gating_scores, dim=-1)   # [b, n, num]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [b, n, d, num]
        output = torch.sum(gating_weights.unsqueeze(2) * expert_outputs, dim=-1)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, num_experts):
        super(EncoderLayer, self).__init__()
        self.dropout = nn.Dropout(0.1)
        hidden_size = 768
        self.norm = nn.LayerNorm(hidden_size)
        self.attn_text = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.attn_image = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.activation = nn.ReLU()
        self.linear_text1 = nn.Linear(hidden_size, hidden_size)
        self.linear_text2 = nn.Linear(hidden_size, hidden_size)
        self.linear_image1 = nn.Linear(hidden_size, hidden_size)
        self.linear_image2 = nn.Linear(hidden_size, hidden_size)

        self.adapter_text1 = MoBA(num_experts, hidden_size)
        self.adapter_text2 = MoBA(num_experts, hidden_size)
        self.adapter_image1 = MoBA(num_experts, hidden_size)
        self.adapter_image2 = MoBA(num_experts, hidden_size)

    def forward(self, text, image):
        text_norm = self.norm(text)
        t_att, _ = self.attn_text(text_norm, text_norm, text_norm)
        t_att = self.dropout(t_att)

        image_norm = self.norm(image)
        v_att, _ = self.attn_image(image_norm, image_norm, image_norm)
        v_att = self.dropout(v_att)
        # pdb.set_trace()
        text_out = text + t_att + self.adapter_text1(image)
        image_out = image + v_att + self.adapter_image1(text)

        text_norm2 = self.norm(text_out)
        t_tmp = self.linear_text1(self.dropout(self.activation(self.linear_text2(self.norm(text_norm2)))))
        text_embeds = text_out + self.dropout(t_tmp)

        image_norm2 = self.norm(image_out)
        i_tmp = self.linear_image1(self.dropout(self.activation(self.linear_image2(self.norm(image_norm2)))))
        image_embeds = image_out + self.dropout(i_tmp)

        text_embeds = self.adapter_text2(image_out) + text_embeds
        image_embeds = self.adapter_image2(text_out) + image_embeds
        return text_embeds, image_embeds


class Encoder(nn.Module):
    def __init__(self, num_experts, n_layers):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList([
            EncoderLayer(num_experts) for _ in range(n_layers)
        ])

    def forward(self, text, image):
        for layer in self.encoders:
            text, image = layer(text, image)
        return text, image


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.model = CLIPModel.from_pretrained(args.model_path)
        self.method = args.fusion
        self.trans = EncoderLayer(args.num_experts)
        self.encoder = Encoder(args.num_experts, args.layers)
        self.classifier_fuse = nn.Linear(args.text_size, args.label_number)
        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1)
        self.concat_linear = nn.Linear(args.text_size * 2, args.text_size)
        self.t1_linear = nn.Linear(args.text_size, args.text_size)
        self.t2_linear = nn.Linear(args.text_size, args.text_size)
        self.i1_linear = nn.Linear(args.text_size, args.text_size)
        self.i2_linear = nn.Linear(args.text_size, args.text_size)
        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        layers_to_freeze = [self.model, self.att,
                            self.classifier_fuse, self.concat_linear,
                            self.t1_linear, self.t2_linear,
                            self.i1_linear, self.i2_linear]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        for name, module in self.trans.named_children():
            if 'adapter' in name or 'attn' in name:
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False
        for encoder_layer in self.encoder.encoders:
            for name, module in encoder_layer.named_children():
                if 'adapter' in name:
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    for param in module.parameters():
                        param.requires_grad = False

    def calculate_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, inputs, labels):
        # embedding
        output = self.model(**inputs, output_attentions=True)
        text_features = output['text_model_output']['last_hidden_state']
        image_features = output['vision_model_output']['last_hidden_state']
        text_embeds = self.model.text_projection(text_features)
        image_embeds = self.model.visual_projection(image_features)

        if image_embeds.size(1) > text_embeds.size(1):
            padding_size = image_embeds.size(1) - text_embeds.size(1)
            text_embeds = F.pad(text_embeds, (0, 0, 0, padding_size))
        elif text_embeds.size(1) > image_embeds.size(1):
            padding_size = text_embeds.size(1) - image_embeds.size(1)
            image_embeds = F.pad(image_embeds, (0, 0, 0, padding_size))

        text_embeds, image_embeds = self.trans(text_embeds, image_embeds)
        text_embeds, image_embeds = self.encoder(text_embeds, image_embeds)
        # fusion
        if self.method == 'add':
            text_feature = torch.mean(text_embeds, dim=1)
            image_feature = torch.mean(image_embeds, dim=1)
            fuse_feature = image_feature + text_feature
        elif self.method == 'concat':
            fuse = torch.concat([image_embeds, text_embeds], dim=-1)
            fuse_feature = torch.mean(fuse, dim=1)
            fuse_feature = self.concat_linear(fuse_feature)
        elif self.method == 'gate':
            a_t = self.softmax(self.t2_linear(self.active(self.t1_linear(text_embeds))))
            a_i = self.softmax(self.i2_linear(self.active(self.i1_linear(image_embeds))))
            y_t = torch.mean(a_t * text_embeds, dim=1)
            y_i = torch.mean(a_i * image_embeds, dim=1)
            fuse_feature = y_t + y_i
        elif self.method == 'att':
            text_feature = torch.mean(text_embeds, dim=1)
            image_feature = torch.mean(image_embeds, dim=1)
            text_weight = self.att(text_feature)
            image_weight = self.att(image_feature)
            att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1), dim=-1)
            tw, iw = att.split([1, 1], dim=-1)
            fuse_feature = tw.squeeze(1) * text_feature + iw.squeeze(1) * image_feature
        # classifier
        logits_fuse = self.classifier_fuse(fuse_feature)
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        outputs = (fuse_score, )
        if labels is not None:
            loss = self.loss_fct(logits_fuse, labels)
            outputs = (loss,) + outputs
        return outputs
