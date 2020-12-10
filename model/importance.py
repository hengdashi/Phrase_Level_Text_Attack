import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


# return units masked with UNK at each position in the sequence
def _get_unk_masked(units):
  len_text = len(units)
  masked_units = []
  for i in range(len_text - 1):
    masked_units.append(units[0:i] + ['[UNK]'] + units[i + 1:])
  
  # list of masked basic units
  return masked_units

def get_important_scores(
    entry,
    tokenizer,
    target_model,
    orig_label,
    orig_logits,
    orig_probs,
    batch_size=1,
    max_length=512
):

  masked_units = _get_unk_masked(entry['phrases'])
  texts = [' '.join(units) for units in masked_units]  # list of text of masked units
  
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
  encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_token_type_ids=False, return_tensors='pt')
    
  eval_data = TensorDataset(encodings['input_ids'], encodings['attention_mask'])

  # Run prediction for full data
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
  leave_1_probs = []
  
  target_model.eval() #make sure in inference stage
  
  with torch.no_grad():
    for batch in eval_dataloader:
      input_ids = batch[0].to(device)      # input ids
      attention_mask = batch[1].to(device) # attention mask
  
      leave_1_prob_batch = target_model(input_ids, attention_mask=attention_mask)[0]
      leave_1_probs.append(leave_1_prob_batch)
  
  leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
  leave_1_probs = torch.softmax(leave_1_probs, -1)
  leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
  import_scores = (orig_logits
                  - leave_1_probs[:, orig_label] # how the probability of original label decreases
                  +
                  (leave_1_probs_argmax != orig_label).float() # new label not equal to original label
                  * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                  ).data.cpu().numpy()           # probability of changed label

  # more importantif can change label
  return import_scores
