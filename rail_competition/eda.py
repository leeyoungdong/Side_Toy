'''
미리 모든 데이터셋을 프로세싱하여 사용.
'''
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def concat_five(df1, df2, df3, df4, df5):
    """
    주어진 다섯 개의 DataFrame을 수직으로 연결(concat)한다.
    
    Parameters:
    - df1, df2, df3, df4, df5: 연결할 DataFrame들
    
    Returns:
    - DataFrame: 연결된 결과 DataFrame
    """
    return pd.concat([df1, df2, df3, df4, df5], axis=0)

def prepare_data_right(filename):
    data = filename
    X = data.drop(columns=["YL_M1_B1_W1", "YR_M1_B1_W1", "YL_M1_B1_W2", "YR_M1_B1_W2"])
    y = data[["YL_M1_B1_W2", "YR_M1_B1_W2"]]

    return X, y

def prepare_data_left(filename):
    data = filename
    X = data.drop(columns=["YL_M1_B1_W1", "YR_M1_B1_W1", "YL_M1_B1_W2", "YR_M1_B1_W2"])
    y = data[["YL_M1_B1_W1", "YR_M1_B1_W1"]]

    return X, y

def prepare_data(data, target_columns):
    """
    입력 데이터에서 타겟 열을 제외하고 X를 추출하고, 타겟 열을 y로 추출한다.
    또한, X에 대한 스케일링을 수행한다.
    
    Parameters:
    - data: 전처리할 입력 데이터 (DataFrame)
    - target_columns: 타겟 변수를 포함하는 열 이름의 리스트
    
    Returns:
    - X: 스케일링된 입력 변수
    - y: 타겟 변수
    """
    X = data.drop(columns=["YL_M1_B1_W1", "YR_M1_B1_W1", "YL_M1_B1_W2", "YR_M1_B1_W2"])
    y = data[target_columns]

    # 데이터 스케일링
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    return X, y

def reshape_data(X, y, N_TIMESTEPS):
    """
    주어진 시계열 데이터를 N_TIMESTEPS 길이의 연속된 시퀀스로 변환한다.
    
    Parameters:
    - X: 입력 변수
    - y: 타겟 변수
    - N_TIMESTEPS: 시퀀스의 길이
    
    Returns:
    - X_list: 변환된 입력 변수의 배열
    - y_list: 각 시퀀스에 해당하는 타겟 변수의 배열
    """
    X_list, y_list = [], []
    X_padded = np.vstack([np.zeros((N_TIMESTEPS, X.shape[1])), X])

    for i in range(len(X_padded) - N_TIMESTEPS):
        X_list.append(X_padded[i:i+N_TIMESTEPS])
        y_list.append(y.iloc[i])

    return np.array(X_list), np.array(y_list)

def getdata():
    """
    CSV 파일들을 읽어들여 train/test 데이터로 분리하고, 각 rawtype에 따라 데이터를 병합한다.

    Returns:
    - data_c: 'c' 유형의 훈련 데이터
    - data_s: 's' 유형의 훈련 데이터
    - data_c_test: 'c' 유형의 테스트 데이터
    - data_s_test: 's' 유형의 테스트 데이터
    """
    # Data loading phase
    raw_data_files_c = ['data_c100.csv', 'data_c30.csv', 'data_c40.csv', 'data_c50.csv', 'data_c70.csv']
    raw_data_files_s = ['data_s100.csv', 'data_s30.csv', 'data_s40.csv', 'data_s50.csv', 'data_s70.csv']
    
    lane_data_c = pd.read_csv('./data/lane_data_c.csv')
    lane_data_s = pd.read_csv('./data/lane_data_s.csv')

    data_c_list, data_s_list, data_c_test_list, data_s_test_list = [], [], [], []
    
    rawtypes = [100, 30, 40, 50, 70]
    
    for file_c, file_s, rawtype in zip(raw_data_files_c, raw_data_files_s, rawtypes):
        data_c_temp = pd.read_csv('./data/' + file_c)
        data_s_temp = pd.read_csv('./data/' + file_s)

        data_c_merge = pd.merge(data_c_temp, lane_data_c)[:-1999]
        data_s_merge = pd.merge(data_s_temp, lane_data_s)[:-1999]
        
        data_c_merge_test = pd.merge(data_c_temp, lane_data_c)[-1999:]
        data_s_merge_test = pd.merge(data_s_temp, lane_data_s)[-1999:]

        data_c_merge['rawtype'] = rawtype
        data_s_merge['rawtype'] = rawtype
        data_c_merge_test['rawtype'] = rawtype
        data_s_merge_test['rawtype'] = rawtype

        data_c_list.append(data_c_merge)
        data_s_list.append(data_s_merge)
        data_c_test_list.append(data_c_merge_test)
        data_s_test_list.append(data_s_merge_test)

    data_c = concat_five(*data_c_list)
    data_s = concat_five(*data_s_list)
    data_c_test = concat_five(*data_c_test_list)
    data_s_test = concat_five(*data_s_test_list)

    for dataset in [data_c, data_s, data_c_test, data_s_test]:
        dataset['./data/Distance'] = (dataset['./data/Distance'] * 4).astype(int)
        dataset.reset_index(inplace=True, drop=True)
    
    return data_c, data_s, data_c_test, data_s_test

def get_mean_std_tf(df):
    """
    주어진 데이터 프레임(df)에 대한 특정 열들의 평균 및 표준편차를 계산한다.

    Parameters:
    - df: 계산할 데이터 (DataFrame)

    Returns:
    - mean: 각 열의 평균 값을 포함하는 텐서
    - std: 각 열의 표준편차 값을 포함하는 텐서
    """
    mean = tf.zeros(4, dtype=tf.float32)
    std = tf.zeros(4, dtype=tf.float32)
    
    mean = tf.tensor_scatter_nd_update(mean, [[0]], [tf.reduce_mean(df['YL_M1_B1_W1'][0:10001])])
    mean = tf.tensor_scatter_nd_update(mean, [[1]], [tf.reduce_mean(df['YR_M1_B1_W1'][0:10001])])
    mean = tf.tensor_scatter_nd_update(mean, [[2]], [tf.reduce_mean(df['YL_M1_B1_W2'][0:10001])])
    mean = tf.tensor_scatter_nd_update(mean, [[3]], [tf.reduce_mean(df['YR_M1_B1_W2'][0:10001])])
    
    std = tf.tensor_scatter_nd_update(std, [[0]], [tf.math.reduce_std(df['YL_M1_B1_W1'][0:10001])])
    std = tf.tensor_scatter_nd_update(std, [[1]], [tf.math.reduce_std(df['YR_M1_B1_W1'][0:10001])])
    std = tf.tensor_scatter_nd_update(std, [[2]], [tf.math.reduce_std(df['YL_M1_B1_W2'][0:10001])])
    std = tf.tensor_scatter_nd_update(std, [[3]], [tf.math.reduce_std(df['YR_M1_B1_W2'][0:10001])])
    
    return mean, std

if __name__ == '__main__':
    print('test1')