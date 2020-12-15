from pprint import pprint

def evaluate(entries, num_pred_failures):
    
  attack_success = 0

  total_q = 0
  total_word_changes = 0
  total_words = 0
  total_phrase_changes = 0
  total_phrases = 0
  query_num = 0
  total_phrase_len = 0
  num_total_entry = len(entries)
  
  for entry in entries:
    if entry['success']:
      attack_success += 1
    total_word_changes += entry['word_changes']
    total_phrase_changes += entry['phrase_changes']
    
    total_words += entry['word_num']
    total_phrases += entry['phrase_num']
    
    query_num += entry['query_num']
    total_phrase_len += entry['phrase_len']
  
  word_change_rate = total_word_changes / total_words
  word_per_seq = total_word_changes / len(entries)

  phrase_change_rate = total_phrase_changes / total_phrases
  phrase_per_seq = total_phrase_changes / len(entries)
                 
  original_acc = 1.0 - num_pred_failures / num_total_entry
  after_atk_acc = 1.0 - (num_pred_failures + attack_success) / num_total_entry
    
  success_rate = attack_success / (num_total_entry - num_pred_failures)

  query_per_attack = query_num / len(entries)
  
  mean_phrase_len = total_phrase_len / total_phrases

  print()
  print('acc/aft-atk-acc: {:.6f}/ {:.6f}, query-per-attack: {:.4f}, success-rate: {:.4f}'.format(original_acc, after_atk_acc, query_per_attack, success_rate))
  print('word-changed-per-attack: {:.4f}, phrase-changed-per-attack: {:.4f}'.format(word_per_seq, phrase_per_seq))
  print('word-changed-rate: {:.4f}, phrase-changed-rate: {:.4f}'.format(word_change_rate, phrase_change_rate))
  print('mean-phrase-length: {:.4f}'.format(mean_phrase_len))
  print()
          