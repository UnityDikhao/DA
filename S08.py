from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

df = pd.read_csv("Groceries_dataset.csv")
t = []
df = df.sample(50)
print(df)
for i in range(0, len(df)):
    t.append(
        [str(df.values[i, j]) for j in range(0, 2) if str(df.values[i, j] != "nan")]
    )
print(t)
te = TransactionEncoder()

t_array = te.fit(t).transform(t)
print(t_array)

df = pd.DataFrame(t_array,columns=te.columns_)
print(df)

freq = apriori(df,min_support=0.005,use_colnames=True)
print(freq)

rules = association_rules(freq,metric='support',min_threshold=0.05)

rules = rules.sort_values(["support","confidence"],ascending=[False,False])

print(rules)

