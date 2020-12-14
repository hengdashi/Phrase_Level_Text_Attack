import numpy as np

embedding_path = './data/sim_mat/counter-fitted-vectors.txt'

word_ids = {}
embeddings = []

with open(embedding_path, 'r') as ifile:
    for i, line in enumerate(ifile):
        line_tokens = line.strip().split()
        word_ids[line_tokens[0]] = i + 1
        embedding = np.array(line_tokens[1:], dtype='float32')
        #norm = np.linalg.norm(embedding)
        #embeddings.append(embedding / norm)
        embeddings.append(embedding)

        if i % 10000 == 0:
            print(i)

embeddings = np.array(embeddings)

print('save embedding matrix & word id')
np.save('./data/sim_mat/embeddings_cf.npy', embeddings)
np.save('./data/sim_mat/word_id.npy', word_ids)
