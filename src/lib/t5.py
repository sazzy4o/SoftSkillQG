# Based on https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py
import copy
import warnings
import torch
import torch.distributed as dist

from dataclasses import dataclass
from pathlib import Path
from torch import nn, tensor
from torch.nn import CrossEntropyLoss,MSELoss,BCELoss
from typing import Optional

from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, BaseModelOutput, Seq2SeqLMOutput, T5Config
from transformers.generation.utils import BeamSearchDecoderOnlyOutput, SampleEncoderDecoderOutput, BeamSearchEncoderDecoderOutput,StoppingCriteriaList,LogitsProcessorList,validate_stopping_criteria

# Based on https://github.com/huggingface/transformers/blob/426b96230a71f3c6e4decabae131c6e4f8bf4f5c/src/transformers/pytorch_utils.py#L24
def torch_int_div(tensor1, tensor2):
    """
    A function that performs integer division across different versions of PyTorch.
    """
    return torch.div(tensor1, tensor2, rounding_mode="floor")

@dataclass
class Seq2SeqLMConfidenceOutput(Seq2SeqLMOutput):
    # confidence_logits: torch.FloatTensor = None
    pass
    
@dataclass
class SampleEncoderDecoderConfidenceOutput(SampleEncoderDecoderOutput):
    # confidence: torch.FloatTensor = None
    probs: torch.FloatTensor = None

# class T5CConfig(T5Config):
#     model_type = "t5"
#     keys_to_ignore_at_inference = ["past_key_values"]
#     attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

#     def __init__(
#         self,
#         vocab_size=32128,
#         lambda_value=0.001,
#         d_model=512,
#         d_kv=64,
#         d_ff=2048,
#         num_layers=6,
#         num_decoder_layers=None,
#         num_heads=8,
#         relative_attention_num_buckets=32,
#         dropout_rate=0.1,
#         layer_norm_epsilon=1e-6,
#         initializer_factor=1.0,
#         feed_forward_proj="relu",
#         is_encoder_decoder=True,
#         use_cache=True,
#         pad_token_id=0,
#         eos_token_id=1,
#         **kwargs
#     ):
#         self.vocab_size = vocab_size
#         self.lambda_value = lambda_value
#         self.d_model = d_model
#         self.d_kv = d_kv
#         self.d_ff = d_ff
#         self.num_layers = num_layers
#         self.num_decoder_layers = (
#             num_decoder_layers if num_decoder_layers is not None else self.num_layers
#         )  # default = symmetry
#         self.num_heads = num_heads
#         self.relative_attention_num_buckets = relative_attention_num_buckets
#         self.dropout_rate = dropout_rate
#         self.layer_norm_epsilon = layer_norm_epsilon
#         self.initializer_factor = initializer_factor
#         self.feed_forward_proj = feed_forward_proj
#         self.use_cache = use_cache
#         super().__init__(
#             pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id,
#             is_encoder_decoder=is_encoder_decoder,
#             **kwargs,
#         )

@dataclass
class BeamSearchEncoderDecoderOutputWithConfidence(BeamSearchEncoderDecoderOutput):
    # confidence: torch.FloatTensor = None
    probs: torch.FloatTensor = None

class T5ForConditionalGenerationWithConfidence(T5ForConditionalGeneration):

    def __init__(self, config,lambda_value=0.001):
        super().__init__(config)
        self.model_dim = config.d_model
        # self.lambda_value = lambda_value
        # print('Lambda value(training only):',lambda_value)

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.confidence_head = nn.Linear(config.d_model, 1)
        # confidence_model_path = Path(config.name_or_path)/'confidence.pt'
        # if confidence_model_path.exists():
        #     print('Loading confience model from checkpoint...')
        #     self.confidence_head.load_state_dict(torch.load(confidence_model_path))

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def save_pretrained(
        self,
        save_directory,
        save_config = True,
        state_dict = None,
        save_function = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(save_directory,save_config,state_dict,save_function,push_to_hub,**kwargs)
        # confidence_model_path = Path(save_directory)/'confidence.pt'
        # torch.save(self.confidence_head.state_dict(), confidence_model_path)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn('__HEAD_MASK_WARNING_MSG', FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        # confidence_logits = self.confidence_head(sequence_output)
        # sig = nn.Sigmoid()
        # confidence_probs = sig(confidence_logits)
        # print(lm_logits.shape)
        # print(confidence_logits.shape)
        # print(labels)
        loss = None
        if labels is not None:
            # with torch.no_grad():
            #     # selector = (labels==-100).detach()
            #     # print('1', flush=True)
            #     shift_preds = torch.argmax(lm_logits.detach(),2)
            #     # print('2',shift_preds.shape, flush=True)
            #     conf_labels = (shift_preds == labels).float()
                # print('3',conf_labels.shape, flush=True)
            # conf_selected = confidence_logits[selector]
            # print('4', flush=True)
                # conf_labels[labels==-100] = 0
            #     mask = torch.ones(conf_labels.size())
            #     conf_labels[labels==-100] = 0
            # confidence_logits = confidence_logits*mask
                # ! TODO: propigate -100 
            # print(confidence_logits[labels==-100])
            # weights = (labels!=-100).float()
            # print('Weight shape:',weights.shape, flush=True)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # print('5',lm_logits.shape, flush=True)
            # conf_loss_fct = BCELoss(weight=weights.view(-1))
            # print('6',labels.shape, flush=True)
            lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # print('CL shape:',confidence_probs.shape, flush=True)
            # print('CLa shape:',conf_labels.shape, flush=True)
            # print('Shape check',confidence_probs.view(-1).shape,conf_labels.view(-1).shape, flush=True)
            # print(confidence_probs, flush=True)
            # print(conf_labels, flush=True)
            # conf_loss = conf_loss_fct(confidence_probs.view(-1), conf_labels.view(-1))
            # loss = lm_loss
            # print('Hit:', flush=True)
            loss = lm_loss
            # loss = lm_loss+self.lambda_value*conf_loss
            # print('LM:',lm_loss)
            # print('Conf:',conf_loss)
            # print('Loss:',loss)
            # print('Fin:',loss.shape, flush=True)
            # Original TO DO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMConfidenceOutput(
            loss=loss,
            logits=lm_logits,
            # confidence_logits=confidence_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    # Based on https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer,
        logits_processor,
        stopping_criteria,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ):
        # print('Beam search!')
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        # conf_values = []

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # print('Raw shape',next_token_logits.shape)
            # next_token_confidence = outputs.confidence_logits
            # print('Conf shape',next_token_confidence.shape)
            # print('Conf shape',next_token_confidence)
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            # print('Next pre',next_token_scores.shape)
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            # conf_view = next_token_confidence.view(batch_size, num_beams)
            # conf_values.append(torch.max(conf_view,1)[0])
            # print('Next:',next_token_scores.shape)
            # print('Conf view', conf_view.shape)
            # print('Max shape', torch.max(conf_view,1)[0].shape)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            # print(next_tokens.shape)
            # print('Toks:',next_tokens)
            # ! TODO: Arg partition

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # print('Ind shape',next_indices.shape)
            # print('Ind:',next_indices)

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            # print('Idx',beam_idx)

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            # print('Conf len',len(conf_values))
            # print('Conf len+',[x.shape for x in conf_values],flush=True)
            # print('Hit 0',flush=True)
            return BeamSearchEncoderDecoderOutputWithConfidence(
                sequences=sequence_outputs["sequences"],
                # confidence=torch.stack(conf_values).T,
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=beam_indices,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return sequence_outputs["sequences"]
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ):

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        # confs = []
        softmax_probs = [torch.zeros(input_ids.shape[0],dtype=torch.float32).to(input_ids.device)]
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            # next_token_confidence = outputs.confidence_logits
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # print('Hit 0',flush=True)
            # print(next_tokens.detach().to('cpu').numpy(),flush=True)
            # print('next_tokens shape',next_tokens.shape,flush=True)
            # print('next_token_confidence shape',next_token_confidence.shape,flush=True)
            # confs.append(next_token_confidence.view(-1).detach())
            # print('probs shape',probs.shape,flush=True)
            softmax_probs.append(torch.index_select(probs,1,next_tokens)[0].detach())

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        # print('Hit 1',flush=True)
        if return_dict_in_generate:
            return SampleEncoderDecoderConfidenceOutput(
                sequences=input_ids,
                scores=scores,
                # confidence=torch.stack(confs).T,
                probs=torch.stack(softmax_probs).T,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids