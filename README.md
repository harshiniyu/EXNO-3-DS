
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
# reg no:212223240050
# name:Harshini Y
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
  ![Screenshot 2025-04-22 105058](https://github.com/user-attachments/assets/66745ee2-5865-401a-a3b8-4ac9d326c81d)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2025-04-22 105156](https://github.com/user-attachments/assets/32574a5c-81f6-44a3-a376-63ead5fa5e7b)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2025-04-22 105428](https://github.com/user-attachments/assets/2416b0c7-8fb3-4c12-b5df-3ebf6814ecbd)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2025-04-22 105453](https://github.com/user-attachments/assets/2aad9667-501b-484f-b6b3-8fb547330ead)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2025-04-22 105619](https://github.com/user-attachments/assets/3cf146b7-09f5-44a3-bb89-e2948c8c570f)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-04-22 105646](https://github.com/user-attachments/assets/06eac686-b7d9-4274-8104-91eb2f9ab340)

```
pip install --upgrade category_encoders
```

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![Screenshot 2025-04-22 105741](https://github.com/user-attachments/assets/23eb7915-a0b7-47f1-86ba-cf66b899e482)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![Screenshot 2025-04-22 105824](https://github.com/user-attachments/assets/269e7a41-7fee-424c-b7e6-017b82bea863)

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![Screenshot 2025-04-22 105852](https://github.com/user-attachments/assets/459fb8d7-9363-4956-b14f-5b0a20da7886)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![Screenshot 2025-04-22 105949](https://github.com/user-attachments/assets/29233447-71ee-428a-83e1-0bf5b91059ad)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![Screenshot 2025-04-22 110009](https://github.com/user-attachments/assets/1e843bb9-fde1-4b72-b37a-dc5c35663966)

```
df.skew()
```
![Screenshot 2025-04-22 110031](https://github.com/user-attachments/assets/daec97ac-320a-452a-8ad4-ae46de16e612)

```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 110048](https://github.com/user-attachments/assets/a40e9902-d805-4a84-8706-b734110d05ae)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-04-22 110113](https://github.com/user-attachments/assets/7242085a-89b4-4e93-a636-a43e36d5aee3)

```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 110130](https://github.com/user-attachments/assets/9c5fa747-70df-4024-a369-ef6dcdc24020)

```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 110150](https://github.com/user-attachments/assets/f3c66cc0-f736-49d1-bc19-7785e6edf326)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2025-04-22 110209](https://github.com/user-attachments/assets/80d61c3e-536d-41aa-8efc-9194ec38b8f9)

```
df.skew()
```
![Screenshot 2025-04-22 110230](https://github.com/user-attachments/assets/a634a480-9dc9-49d8-8fa0-41cf9c59c8d6)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2025-04-22 110247](https://github.com/user-attachments/assets/0cc7da28-19d1-4dc6-98c7-97f31a5ec283)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![Screenshot 2025-04-22 110311](https://github.com/user-attachments/assets/7bff0cfa-4e7a-4c40-9449-b3bcf300939a)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 110333](https://github.com/user-attachments/assets/7d46e654-2d0e-4a26-b13e-f119d8080589)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```
![Screenshot 2025-04-22 110354](https://github.com/user-attachments/assets/77579aa0-180c-4606-938a-abc34f7ba96d)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 110416](https://github.com/user-attachments/assets/f05e8b4b-b4d5-4f76-90d0-86186cd9a919)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 110435](https://github.com/user-attachments/assets/5f64ac11-d40a-49ee-9215-489f6dba5587)

```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![Screenshot 2025-04-22 110515](https://github.com/user-attachments/assets/652c48be-0243-4810-96c3-076f9eded42b)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![Screenshot 2025-04-22 110542](https://github.com/user-attachments/assets/82217b89-717a-4243-aa89-1745904c0d40)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![Screenshot 2025-04-22 110559](https://github.com/user-attachments/assets/c8c9f42d-8db9-4222-b6a0-7d034de5d498)

# RESULT:

Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
