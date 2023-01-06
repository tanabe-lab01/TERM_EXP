import numpy as np


def corrupt_label(train_labels, noise_ratio=[0.8]):
    # ラベルは0~9にしか対応してない
    # np.random.seed(1)

    # ---------------------------------------------------------------
    #   input:
    #       train_labels:変更前のラベルセット
    #       noise_ratio :各アノテータ―がラベルをランダムに変更する
    #                    (間違えてラベル付けをしてしまう)確率
    #   output:
    #       train_labels:変更後のラベルセット
    #       normal_ind  :正常ラベルを指すインデックス
    #       anotator    :どのアノテータ―がラベル付けを担当したか
    # ---------------------------------------------------------------

    num_annotator = len(noise_ratio)
    label_ = int(len(train_labels)/num_annotator)
    noisy = np.array(noise_ratio) * label_

    normal_flag = []

    anotator = []

    for i in range(num_annotator):
        if noise_ratio[i] == 0:
            normal_flag.extend([0]*label_)

        elif noise_ratio[i] > 0:
            subset = train_labels[i * label_: (i+1) * label_]

            for j in range(len(subset)):
                if np.random.rand() > 1 - noise_ratio[i]:
                    subset[j] = np.random.choice(10)
                    normal_flag.append(1)
                else:
                    subset[j] = train_labels[i * label_ + j]
                    normal_flag.append(0)

            train_labels[i * label_: (i+1) * label_] = subset[:]

        anotator[i * label_: (i+1)*label_] = [i] * label_

    normal_flag.extend([0]*(len(train_labels)-(num_annotator)*label_))
    anotator[num_annotator*label_: len(train_labels)] = [-1] * \
        (len(train_labels) - num_annotator*label_)

    return train_labels, np.array(normal_flag).flatten(), np.array(anotator).flatten()


if __name__ == '__main__':
    true_label = [i for i in range(10)]
    annotator = [i/10 for i in range(5)]
    print(f'annotator       = {annotator}')
    print(f'true_label      = {np.array(true_label).flatten()}')
    ret = corrupt_label(true_label, annotator)
    print(f'corrupted_label = {np.array(ret[0]).flatten()}')
    print(f'normal_index    = {ret[1]}')
    print(f'annotator       = {ret[2]}')
