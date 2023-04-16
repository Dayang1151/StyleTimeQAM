import pandas as pd
# a = pd.read_csv('new_test.csv')
# result = []
# for i in range(len(a['match'])):
#     b = []
#     for j in range(len(a['match'][i][1:-1])):
#         if a['match'][i][1:-1][j] != ',' and a['match'][i][1:-1][j] != ' ':
#             b.append(int(a['match'][i][1:-1][j]))
#     result.append(b)
# df = pd.DataFrame(result)
# df.to_csv('test.csv')
import pandas as pd

# 假设有列表a
a = [[1,2,3,4,5],[1,2,3,4,5]]
# 将list转为dataframe 显然就变成一列了
d = pd.DataFrame(a)
d.to_csv('test.csv',index=False,header=None) # mode表示追加 在追加时会将列名也作为一行进行追加，故header隐藏表头（列名）

