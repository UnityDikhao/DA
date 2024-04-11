import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Reading the dataset
df = pd.read_csv("Market_Basket_Optimisation.csv")

# Empty list to store transactions
t = []

# Sample 50 rows from the dataset
df = df.sample(50)

# Iterating through each row of the dataframe
for i in range(0, len(df)):
    # Extracting items from each row and appending to list t
    t.append(
        [str(df.values[i, j]) for j in range(0, 20) if str(df.values[i, j] != "nan")]
    )

# Printing the list of transactions
print(t)

# Initializing TransactionEncoder
te = TransactionEncoder()

# Transforming the list of transactions into a transaction matrix
te_array = te.fit(t).transform(t)

# Creating a new dataframe from the transaction matrix
df = pd.DataFrame(te_array, columns=te.columns_)

# Printing the dataframe
print(df)

# Finding frequent item sets using Apriori algorithm with minimum support threshold of 0.005
freq_items = apriori(df, min_support=0.005, use_colnames=True)

# Printing frequent item sets
print(freq_items)

# Generating association rules from frequent item sets with minimum support threshold of 0.05
rules = association_rules(freq_items, metric="support", min_threshold=0.05)

# Sorting the rules based on support and confidence in descending order
rules = rules.sort_values(["support", "confidence"], ascending=[False, False])

# Printing the association rules
print(rules)
