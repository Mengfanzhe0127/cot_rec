from transformers import Trainer
import os
import json

class CustomTrainer(Trainer):
    """
    each training step record input data
    """
    def __init__(self, *args, **kwargs):
        self.steps_to_log = kwargs.pop("steps_to_log", None)
        super().__init__(*args, **kwargs)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        execute a training step and record input data
        
        args:
            model (`nn.Module`): the model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`): the dictionary passed to the model.
            
        returns:
            `torch.Tensor`: the loss value.
        """
        should_log = (self.steps_to_log is None or 
                      self.state.global_step in self.steps_to_log)
        if should_log:
            try:
                rank = getattr(self.args, "local_rank", 0)
                if rank == -1:  # -1表示没有使用分布式训练
                    rank = 0
                
                log_dir = os.path.join(self.args.output_dir, "batch_logs")
                os.makedirs(log_dir, exist_ok=True)
                
                step_file = os.path.join(
                    log_dir, 
                    f"step_{self.state.global_step}_rank_{rank}_batch.json"
                )
            
                serializable_data = {}
            
                if "input_ids" in inputs and hasattr(self, "processing_class") and self.processing_class is not None:
                    input_ids = inputs["input_ids"]

                    decoded_texts = []
                    batch_size = input_ids.shape[0]
                
                    for i in range(batch_size):
                        try:
                            decoded_text = self.processing_class.decode(input_ids[i], skip_special_tokens=False)
                            decoded_texts.append(decoded_text)
                        except Exception as e:
                            decoded_texts.append(f"decode error: {str(e)}")
                
                    serializable_data["input_ids_decoded"] = decoded_texts
            
                serializable_data["batch_info"] = {
                    "global_step": self.state.global_step,
                    "epoch": self.state.epoch,
                    "rank": rank,
                }
                
                with open(step_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
                print(f"GPU {rank}: saved step {self.state.global_step} batch data to {step_file}")
            
            except Exception as e:
                print(f"GPU {rank}: error when saving step {self.state.global_step} batch data: {str(e)}")

        return super().training_step(model, inputs)