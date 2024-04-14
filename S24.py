import pandas as pd
import matplotlib.pyplot as pltO

data = pd.read_csv("INvideos.csv")

data.dropna(inplace=True)

total_views = data['views'].sum()
total_likes = data['likes'].sum()
total_dislikes = data['dislikes'].sum()
total_comments = data['comment_count'].sum()

print("Total views:", total_views)
print("Total likes:", total_likes)
print("Total dislikes:", total_dislikes)
print("Total comment count:", total_comments)

