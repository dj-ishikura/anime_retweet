import peft
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel
from torch import FloatTensor, LongTensor
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutput,
)
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lora_r: int,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        backbone: PreTrainedModel = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None,
        )

        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            lora_dropout=0.1,
            inference_mode=False,
        )
        self.backbone: PeftModel = peft.get_peft_model(backbone, self.peft_config)

        if gradient_checkpointing:
            self.backbone.enable_input_require_grads()
            self.backbone.gradient_checkpointing_enable()

        hidden_size: int = self.backbone.config.hidden_size
        self.classifiers_emotion = nn.ModuleList([
            nn.Linear(hidden_size, 4) for _ in range(8)
        ])
        self.classifier_polarity = nn.Linear(hidden_size, 5)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor = None,
        labels: LongTensor = None,
    ) -> SequenceClassifierOutput:
        outputs: BaseModelOutputWithPast = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        seq_length: LongTensor = attention_mask.sum(dim=1)
        eos_hidden_states: FloatTensor = outputs.last_hidden_state[
            torch.arange(
                seq_length.size(0),
                device=outputs.last_hidden_state.device,
            ),
            seq_length - 1,
        ]

        logits_list = []
        # print(f'self.classifiers_emotion : \n {self.classifiers_emotion}')

        for classifier in self.classifiers_emotion:
            logits = classifier(eos_hidden_states)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            logits_list.append(predictions)
            # print(f'Logits: \n{logits}\nProbabilities: \n{probabilities}\nPredictions: \n{predictions}')

        
        logits_polarity = self.classifier_polarity(eos_hidden_states)
        probabilities = F.softmax(logits_polarity, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        # print(f'logits_polarity: \n{logits_polarity}\nProbabilities: \n{probabilities}\nPredictions: \n{predictions}')
        logits_list.append(predictions)

        # ラベルが提供されている場合のみ損失を計算
        # print(f'labels : \n {labels}')
        # print(f'logits_list : \n {logits_list}')
        if labels is not None:
            loss = 0
            # 感情カテゴリーの損失を計算
            for i, logits in enumerate(logits_list[:-1]):  # 最後の要素（感情極性）を除外
                # print(f"Logits size: {logits.size()}, Labels size: {labels[:, i].size()}")
                # print(f"Logits dtype: {logits.dtype}, Labels dtype: {labels[:, i].dtype}")
                loss += self.loss_fn(logits.float(), labels[:, i])

            # 感情極性の損失を計算
            loss += self.loss_fn(logits_list[-1].float(), labels[:, -1])

            # logits_listの各テンソルをバッチ次元に沿って結合
            logits_combined = torch.stack(logits_list, dim=-1)

            # 結合したテンソルを転置
            logits_transposed = logits_combined.t()

            # print(f"Combined logits shape: {logits_combined.shape}")
            # print(f"Transposed logits shape: {logits_transposed.shape}")
            return SequenceClassifierOutput(loss=loss, logits=logits_list)
        else:
            return SequenceClassifierOutput(logits=torch.cat(logits_list, dim=-1))


    def write_trainable_params(self) -> None:
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        percentage = 100 * trainable_params / all_param
        all_param /= 1000000000
        trainable_params /= 1_000_000

        print(
            f"trainable params: {trainable_params:.2f}M || "
            f"all params: {all_param:.2f}B || "
            f"trainable%: {percentage:.4f}"
        )

    def clone_state_dict(self) -> dict:
        state_dict = {
            "backbone": peft.get_peft_model_state_dict(self.backbone),
            "classifiers_emotion": [classifier.state_dict() for classifier in self.classifiers_emotion],
            "classifier_polarity": self.classifier_polarity.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        peft.set_peft_model_state_dict(self.backbone, state_dict["backbone"])

        for classifier, classifier_state in zip(self.classifiers_emotion, state_dict["classifiers_emotion"]):
            classifier.load_state_dict(classifier_state)

        self.classifier_polarity.load_state_dict(state_dict["classifier_polarity"])


