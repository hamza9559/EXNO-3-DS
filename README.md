## EXNO-3-DS
# NAME: HAMZA FAROOQUE
# REG.NO: 212223040054
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
```
import pandas as pd
import numpy as np
df= pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/15c8d9cc-e0e9-49a8-a91f-c5b93874479e)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/c932ab47-2891-4473-858b-9dd459dd48ab)

```
le= LabelEncoder()
dfc= df.copy()
dfc
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/1910d780-59a1-41a9-9638-f6d8b1d87e3d)

```
dfc= df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/20fcdfb6-f078-4897-8507-0f272e2c707a)

# ONEHOT ENCODER 

```
from sklearn.preprocessing import OneHotEncoder
# Use sparse_output instead of sparse for newer versions of scikit-learn
ohe= OneHotEncoder(sparse_output=False)
df2=df.copy()
df2
```
![image](https://github.com/user-attachments/assets/1b72493d-9c28-4768-9069-e8aed97cd11b)

```
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2= pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/269a51bd-c45d-4606-aa54-99372aac891d)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/a6c972ee-a821-4fa6-96b8-ee0621f5ff88)

```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/a566cf92-5feb-4d13-ac81-472dc7fc82e1)

```
from category_encoders import BinaryEncoder
df= pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/74eeb888-84a0-46df-909d-55e50cfdf6c0)

```
be= BinaryEncoder()
nd= be.fit_transform(df['Ord_2'])
dfb= pd.concat([df,nd],axis=1)
dfb1= df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/c7867172-5d0d-4fbd-a935-44c45c3c186f)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/1dc62eca-1d27-4007-94d5-b5e9002f82cd)

```
import pandas as pd
from scipy import stats
import numpy as np
df= pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/291d01ab-2248-4af3-b1a3-ffa0aa70e1f0)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/8a11d74e-5920-4e2d-91e1-517ee7e0edd2)

```
df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])
df['Moderate Positive Skew']=np.reciprocal(df['Moderate Negative Skew'])
df['Highly Negative Skew']= np.sqrt(df['Highly Negative Skew'])
df.skew()
```
![image](https://github.com/user-attachments/assets/663ce383-ac94-4f1f-8832-0a89de300060)

```
df['Highly Positive Skew_boxcox'], parameters= stats.boxcox(df['Highly Positive Skew'])
df['Moderate Negative Skew_yeojohnson'], parameters= stats.yeojohnson(df['Moderate Positive Skew'])
df
```
![image](https://github.com/user-attachments/assets/7c8ee17f-b7c5-4b82-90ee-a3167195e919)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/887824ad-62bb-455e-91db-2cfb54532ac1)

# POWER TRANSFORMATION

```
from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution='normal')
df['Moderate Negative Skew']= qt.fit_transform(df[['Moderate Negative Skew']])
df
```
![image](https://github.com/user-attachments/assets/73419402-c667-4d0f-96f5-03e3b8a8af67)

```
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/88dab7c7-a110-4b32-b4f1-6938c85a8c53)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3579821d-38e0-469f-a40b-714d229b8023)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/17ba70a1-ca6b-4573-9771-e28cb635fb0c)



# RESULT:
       Thus,  Feature Encoding and Transformation process is performed on the given data.

       
