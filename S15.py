from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules
import pandas as pd 

transaction=[
    {'company' :['tata','mg','kia','hyundia'],'model':['nexon','astor','seltos','creta'],'year':[2017,2021,2019,2015]}
]
te = TransactionEncoder()

te_array = te.fit(transaction).transform(transaction)
df = pd.DataFrame(te_array,columns=te.columns_)
print(df)

freq_items = apriori(df,min_support=0.005,use_colnames=True)
print(freq_items)

rules = association_rules(freq_items,metric ='support',min_threshold=0.05)

rules = rules.sort_values(['support','confidence'],ascending=[False,False])

print(rules)