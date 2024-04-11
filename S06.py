import pandas as pd  # Importing pandas library for data manipulation
from mlxtend.preprocessing import TransactionEncoder  # Importing TransactionEncoder for preprocessing
from mlxtend.frequent_patterns import apriori, association_rules  # Importing apriori and association_rules functions

# List of transactions where each transaction is represented as a list of items
transactions = [['bread', 'milk'],
                ['bread', 'diaper', 'Beer', 'eggs'],
                ['milk', 'diaper', 'Beer', 'Coke'],
                ['bread', 'milk', 'diaper', 'Beer'],
                ['bread', 'milk', 'diaper', 'Beer']]

te = TransactionEncoder()  # Initializing TransactionEncoder
te_array = te.fit(transactions).transform(transactions)  # Transforming transaction data into binary format
df = pd.DataFrame(te_array, columns=te.columns_)  # Creating DataFrame from transformed transaction data

print(df)  # Printing the DataFrame showing binary representation of transactions

# Finding frequent itemsets using Apriori algorithm with minimum support threshold of 0.5
# and using column names in the output
freq_items = apriori(df, min_support=0.5, use_colnames=True)

print(freq_items)  # Printing frequent itemsets along with their support values

# Generating association rules from frequent itemsets using support metric
# with a minimum support threshold of 0.05
rules = association_rules(freq_items, metric='support', min_threshold=0.05)

# Sorting association rules DataFrame based on support and confidence values in descending order
rules = rules.sort_values(['support', 'confidence'], ascending=[False, False])

print(rules)  # Printing association rules sorted by support and confidence
