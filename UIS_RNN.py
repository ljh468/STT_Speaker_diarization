# 학습 및 평가에는 uisrnn 라이브러리에서 제공하는 sample dataset을 사용
# urlretrieve를 통해 url에서 데이터를 받아옴
# https://github.com/google/uis-rnn/blob/master/data/toy_training_data.npz?raw=True
# https://github.com/google/uis-rnn/blob/master/data/toy_testing_data.npz?raw=True
import urllib.request

# 학습 데이터
training_url = 'https://github.com/google/uis-rnn/blob/master/data/toy_training_data.npz?raw=True'
urllib.request.urlretrieve(training_url, './toy_training_data.npz')

# 테스트 데이터
testing_url = 'https://github.com/google/uis-rnn/blob/master/data/toy_testing_data.npz?raw=True'
urllib.request.urlretrieve(testing_url, './toy_testing_data.npz')

import numpy as np
import uisrnn

# 다운로드한 데이터에서 데이터를 받아오고 시퀀스와 라벨로 분리
train_data = np.load('./toy_training_data.npz', allow_pickle=True)
test_data = np.load('./toy_testing_data.npz', allow_pickle=True)

train_sequence = train_data['train_sequence']
train_cluster_id = train_data['train_cluster_id']

test_sequences = test_data['test_sequences'].tolist()
test_cluster_ids = test_data['test_cluster_ids'].tolist()

import easydict

model_args = easydict.EasyDict({'crp_alpha': 1.0,
                                'enable_cuda': True,
                                'observation_dim': 256,
                                'rnn_depth': 1,
                                'rnn_dropout': 0.2,
                                'rnn_hidden_size': 512,
                                'sigma2': None,
                                'transition_bias': None,
                                'verbosity': 2
                                })

training_args = easydict.EasyDict({'batch_size': 10,
                                   'enforce_cluster_id_uniqueness': True,
                                   'grad_max_norm': 5.0,
                                   'learning_rate': 0.001,
                                   'num_permutations': 10,
                                   'optimizer': 'adam',
                                   'regularization_weight': 1e-05,
                                   'sigma_alpha': 1.0,
                                   'sigma_beta': 1.0,
                                   'train_iteration': 5000
                                   })

inference_args = easydict.EasyDict({'batchsize': 100,
                                    'look_ahead': 1,
                                    'test_iteration': 2,
                                    'beam_size': 10})

# 모델 학습
model = uisrnn.UISRNN(model_args)
model.fit(train_sequence, train_cluster_id, training_args)

# 모델 평가
predicted_cluster_ids = []
test_record = []

for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
    predicted_cluster_id = model.predict(test_sequence, inference_args)
    predicted_cluster_ids.append(predicted_cluster_id)
    accuracy = uisrnn.compute_sequence_match_accuracy(test_cluster_id, predicted_cluster_id)
    # 정확성, test_cluster_id의 개수를 튜플로
    test_record.append((accuracy, len(test_cluster_id)))
    print('Ground truth labels : ')
    print(test_cluster_id)
    print('Predicted labels : ')
    print(predicted_cluster_id)
    print('-' *100)

# 최종 결과 출력
output_result = uisrnn.output_result(model_args, training_args, test_record)
print(output_result)