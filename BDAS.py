
# coding: utf-8

# In[1]:


import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('BDAS').getOrCreate()


# In[2]:


dataset = spark.read.csv('Life Expectancy Data.csv', inferSchema=True, header=True)
dataset.printSchema()


# In[3]:


dataset.show()


# In[4]:


print((dataset.count(), len(dataset.columns)))


# In[5]:


dataset.describe().show()


# In[6]:


dataset.dtypes


# In[7]:


dataset.columns


# In[8]:


dataset.head()


# In[17]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


statusArr = np.array(dataset.select('Status').collect())
plt.hist(statusArr)
plt.show()


# In[12]:


yearArr = np.array(dataset.select('Year').collect())
plt.hist(yearArr)
plt.show()


# In[13]:


countryArr = np.array(dataset.select('Country').collect())
plt.hist(countryArr)
plt.show()


# In[14]:


dataset=dataset.toPandas()
sns.pairplot(dataset)


# In[15]:


plt.figure(figsize=(16,16))
cor = sns.heatmap(dataset.corr(), annot = True)
plt.show()


# In[16]:


alpha = 0.7
plt.figure(figsize=(10,25))
sns.countplot(y='Country', data=dataset, alpha=alpha)
plt.title('Data by country')
plt.show()


# In[17]:


plt.figure(figsize=(25,8))
sns.countplot(x='Year', data=dataset, alpha=alpha)
plt.title('Data by year')
plt.axhline(y=150, color='k')
plt.show()


# In[18]:


plt.figure(figsize=(16,7))
bar_gen = sns.barplot(x = 'Country', y = 'Life expectancy ', hue = 'Year',data = dataset)


# In[19]:


dataset.dtypes


# In[5]:


columns_to_drop = ['Population']
select_data = dataset.drop(*columns_to_drop)


# In[6]:


select_data.columns


# In[7]:


print((select_data.count(), len(select_data.columns)))


# In[31]:


print(dataset.isnull().sum())


# In[8]:


from pyspark.sql.functions import isnan, when, count, col

select_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in select_data.columns]).show()


# In[9]:


clean_data=select_data.fillna(0)


# In[10]:


clean_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in clean_data.columns]).show()


# In[11]:


clean_data.dtypes


# In[24]:


clean_data.select("GDP").show()


# In[25]:


import pyspark.sql.functions as f

new=clean_data.select("GDP", f.regexp_replace(f.col("GDP"), "[\$#,]", "").alias("gdp"))
new.show()


# In[23]:


import pyspark.sql.functions as f

construct_data=clean_data.withColumn("gdp",f.regexp_replace(f.col("GDP"), "[\$#,]", ""))
construct_data.dtypes


# In[26]:


from pyspark.sql.types import IntegerType
construct_data = construct_data.withColumn("gdp", construct_data["gdp"].cast(IntegerType()))
construct_data.dtypes


# In[17]:


construct_data.select("gdp").show()


# In[18]:


columns_to_drop = ['GDP']
construct_data = construct_data.drop(*columns_to_drop)


# In[27]:


construct_data.columns


# In[28]:


from  pyspark.sql.functions import abs

construct_data2 = construct_data.withColumn('gdp',abs(construct_data["gdp"]))
construct_data2.select("gdp").show()


# In[29]:


construct_data2.dtypes


# In[30]:


construct_data2.show()


# In[37]:


construct_data2.toPandas().to_csv('data_integration.csv')


# In[38]:


data_integration = spark.read.csv('data_integration.csv', inferSchema=True, header=True)
data_integration.printSchema() 
print((data_integration.count(), len(data_integration.columns)))


# In[39]:


columns_to_drop = ['_c0']
data_integration = data_integration.drop(*columns_to_drop)
data_integration.columns


# In[40]:


data_integration = data_integration.toPandas()

print("country: ", data_integration["Country"].unique())
print("Status: ", data_integration["Status"].unique())


# In[43]:


get_ipython().system('pip install scikit-spark')


# In[46]:


get_ipython().system('pip install sklearn')


# In[47]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data_integration['Country'] = le.fit_transform(data_integration['Country'])
data_integration['Status'] = le.fit_transform(data_integration['Status'])
data_integration.info()


# In[48]:


print("country: ", data_integration["Country"].unique())
print("Status: ", data_integration["Status"].unique())


# In[49]:


data_integration.to_csv("prepared_data.csv")


# In[6]:


prepared_data = spark.read.csv('prepared_data.csv', inferSchema=True, header=True)


# In[7]:


columns_to_drop = ['_c0']
prepared_data = prepared_data.drop(*columns_to_drop)


# In[8]:


prepared_data.printSchema() 
print((prepared_data.count(), len(prepared_data.columns)))


# In[9]:


import six
for i in prepared_data.columns:
    if not(isinstance(prepared_data.select(i).take(1)[0][0], six.string_types)):
        print("Correlation to Life expectancy for ", i, prepared_data.stat.corr('Life expectancy ',i))


# In[10]:


columns_to_drop = ['Country']
reduce_data = prepared_data.drop(*columns_to_drop)
reduce_data.columns


# In[13]:


targetArr = np.array(prepared_data.select('Life expectancy ').collect())
plt.hist(targetArr)
plt.show()


# In[14]:



from pyspark.sql.functions import col
from pyspark.sql.functions import log

project_data = reduce_data.withColumn("log_rate", log(col("Life expectancy ")+1))
project_data.columns


# In[15]:


project_data.select("Life expectancy ","log_rate").show()


# In[16]:


targetArr = np.array(project_data.select('log_rate').collect())
plt.hist(targetArr)
plt.show()


# In[6]:


project_data.toPandas().to_csv('project_data.csv')


# In[2]:


df = spark.read.csv('project_data.csv', inferSchema=True, header=True)
columns_to_drop = ['_c0']
df = df.drop(*columns_to_drop)


# In[6]:


df.printSchema() 
print((df.count(), len(df.columns)))


# In[3]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


# In[4]:


assembler = VectorAssembler(
    inputCols=["Year", "Status", 
               "Adult Mortality","infant deaths","Alcohol","percentage expenditure",
               "Hepatitis B","Measles ", " BMI ","under-five deaths ", "Polio","Total expenditure", "Diphtheria ", " HIV/AIDS", "gdp", " thinness  1-19 years", " thinness 5-9 years", "Income composition of resources", "Schooling","log_rate"],
    outputCol="features")


# In[5]:


output = assembler.transform(df)
output.printSchema()
output.head(1)


# In[10]:


final_df = output.select("features",'Life expectancy ')
final_df.show()


# In[28]:


final_df.describe().show()


# In[11]:


train_data,test_data = final_df.randomSplit([0.7,0.3])


# In[30]:


train_data.describe().show()
test_data.describe().show()


# In[7]:


lr = LinearRegression(labelCol='Life expectancy ')


# In[12]:


lrModel = lr.fit(train_data)


# In[13]:


print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))


# In[34]:


lrModel.coefficients


# In[14]:


test_results = lrModel.evaluate(test_data)


# In[36]:


test_results.residuals.show()


# In[15]:


predictions_lr = lrModel.transform(test_data)
predictions_lr.select("prediction","Life expectancy ","features").show()


# In[38]:


print("RSME: {}".format(test_results.rootMeanSquaredError))


# In[39]:


print("R2: {}".format(test_results.r2))


# In[43]:


import numpy as np 
import matplotlib.pyplot as plt

beta=np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()


# In[18]:


plt.scatter(x=predictions_lr.toPandas()["prediction"],y=predictions_lr.toPandas()["Life expectancy "])
plt.title('Make predictions of Linear Regression')
plt.xlabel('Predicted Life expectancy ')
plt.ylabel('Actual Life expectancy')
plt.show()


# In[21]:


lr_lasso = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=1.0,
                            labelCol='Life expectancy ')


# In[22]:


lr_lassoModel = lr_lasso.fit(train_data)
print("Coefficients(Lasso): {} Intercept(Lasso): {}".format(lr_lassoModel.coefficients,lr_lassoModel.intercept))


# In[23]:


test_results_lasso = lr_lassoModel.evaluate(test_data)
test_results_lasso.residuals.show()


# In[24]:


predictions_lasso = lr_lassoModel.transform(test_data)
predictions_lasso.select("prediction","Life expectancy ","features").show()


# In[48]:


print("RSME(Lasso): {}".format(test_results_lasso.rootMeanSquaredError))


# In[49]:


print("R2(Lasso): {}".format(test_results_lasso.r2))


# In[50]:


beta_lasso=np.sort(lr_lassoModel.coefficients)
plt.plot(beta_lasso)
plt.ylabel('Beta Coefficients (Lasso)')
plt.show()


# In[25]:


plt.scatter(x=predictions_lasso.toPandas()["prediction"],y=predictions_lasso.toPandas()["Life expectancy "])
plt.title('Make predictions of Lasso Regression')
plt.xlabel('Predicted Life expectancy ')
plt.ylabel('Actual Life expectancy ')
plt.show()


# In[26]:


lr_ridge = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0,
                            labelCol='Life expectancy ')


# In[27]:


lr_ridgeModel = lr_ridge.fit(train_data)
print("Coefficients(Ridge): {} Intercept(Ridge): {}".format(lr_ridgeModel.coefficients,lr_ridgeModel.intercept))


# In[28]:


test_results_ridge = lr_ridgeModel.evaluate(test_data)
test_results_ridge.residuals.show()


# In[30]:


predictions_ridge = lr_ridgeModel.transform(test_data)
predictions_ridge.select("prediction","Life expectancy ","features").show()


# In[56]:


print("RSME(Ridge): {}".format(test_results_ridge.rootMeanSquaredError))


# In[57]:


print("R2(Ridge): {}".format(test_results_ridge.r2))


# In[58]:


beta_ridge=np.sort(lr_ridgeModel.coefficients)
plt.plot(beta_ridge)
plt.ylabel('Beta Coefficients (Ridge)')
plt.show()


# In[32]:


plt.scatter(x=predictions_ridge.toPandas()["prediction"],y=predictions_ridge.toPandas()["Life expectancy "])
plt.title('Make predictions of Ridge Regression')
plt.xlabel('Predicted Life expectancy ')
plt.ylabel('Actual Life expectancy')
plt.show()


# In[33]:


lr_ElasticNet = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,
                                 labelCol='Life expectancy ')


# In[34]:


lr_ElasticNetModel = lr_ElasticNet.fit(train_data)
print("Coefficients(Elastic-Net): {} Intercept(Elastic-Net): {}".format(lr_ElasticNetModel.coefficients,
                                                            lr_ElasticNetModel.intercept))


# In[35]:


test_results_ElasticNet = lr_ElasticNetModel.evaluate(test_data)
test_results_ElasticNet.residuals.show()


# In[36]:


predictions_ElasticNet = lr_ElasticNetModel.transform(test_data)
predictions_ElasticNet.select("Life expectancy ","features").show()


# In[37]:


print("RSME(Elastic-Net): {}".format(test_results_ElasticNet.rootMeanSquaredError))


# In[66]:


print("R2(Elastic-Net): {}".format(test_results_ElasticNet.r2))


# In[67]:


beta_ElasticNet=np.sort(lr_ElasticNetModel.coefficients)
plt.plot(beta_ElasticNet)
plt.ylabel('Beta Coefficients (Elastic-Net)')
plt.show()


# In[39]:


plt.scatter(x=predictions_ElasticNet.toPandas()["prediction"],y=predictions_ElasticNet.toPandas()["Life expectancy "])
plt.title('Make predictions of Elastic-Net')
plt.xlabel('Predicted Life expectancy ')
plt.ylabel('Actual Life expectancy ')
plt.show()


# In[71]:


dataset=df.toPandas()
sns.pairplot(dataset)


# In[72]:


dataset.pivot_table('Life expectancy ', index='Year', columns='Status', aggfunc='sum').plot()
plt.title('Life expectancy Per Year From 2000 To 2015 Hue To Status')
plt.ylabel('Life expectancy ')
plt.xlabel('Year')
plt.xlim((dataset.year.min() - 1), (dataset.year.max() + 1))
plt.show()


# In[73]:


plt.figure(figsize=(15,5))
sns.lineplot(x='Year', y='Life expectancy ', data=dataset, color='navy')
plt.axhline(dataset['Life expectancy '].mean(), ls='--', color='red')
plt.title('Life expectancy (by year)')
plt.xlim(2000,2015)
plt.show()


# In[8]:


import six
for i in df.columns:
    if not(isinstance(df.select(i).take(1)[0][0], six.string_types)):
        print("Correlation to log_rate for ", i, df.stat.corr('log_rate',i))


# In[9]:


assembler = VectorAssembler(
    inputCols=["Year", "Status", 
               "Adult Mortality","infant deaths","Alcohol","percentage expenditure",
               "Hepatitis B","Measles ", " BMI ","under-five deaths ", "Polio","Total expenditure", "Diphtheria ", " HIV/AIDS", "gdp", " thinness  1-19 years", " thinness 5-9 years", "Income composition of resources", "Schooling","log_rate"],
    outputCol="features")


# In[11]:


output = assembler.transform(df)
output.printSchema()
output.head(1)


# In[12]:


final_df = output.select("features",'log_rate')
final_df.show()


# In[13]:


final_df.describe().show()


# In[14]:


train_data,test_data = final_df.randomSplit([0.7,0.3])
train_data.describe().show()
test_data.describe().show()


# In[15]:


# Linear Regression
lr = LinearRegression(labelCol='log_rate')
# Lasso Regression
lr_lasso = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=1.0,
                            labelCol='log_rate')
# Ridge Regression
lr_ridge = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0,
                            labelCol='log_rate')
# Elastic-Net
lr_ElasticNet = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,
                                 labelCol='log_rate')
# Fit the model to the data.
lrModel = lr.fit(train_data)
lr_lassoModel = lr_lasso.fit(train_data)
lr_ridgeModel = lr_ridge.fit(train_data)
lr_ElasticNetModel = lr_ElasticNet.fit(train_data)


# In[16]:


print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))
print("\n")
print("Coefficients(Lasso): {} Intercept(Lasso): {}".format(lr_lassoModel.coefficients,lr_lassoModel.intercept))
print("\n")
print("Coefficients(Ridge): {} Intercept(Ridge): {}".format(lr_ridgeModel.coefficients,lr_ridgeModel.intercept))
print("\n")
print("Coefficients(Elastic-Net): {} Intercept(Elastic-Net): {}".format(lr_ElasticNetModel.coefficients,
                                                        lr_ElasticNetModel.intercept))


# In[17]:


test_results = lrModel.evaluate(test_data)
test_results.residuals.show()


# In[18]:


test_results_lasso = lr_lassoModel.evaluate(test_data)
test_results_lasso.residuals.show()


# In[19]:


test_results_ridge = lr_ridgeModel.evaluate(test_data)
test_results_ridge.residuals.show()


# In[20]:


test_results_ElasticNet = lr_ElasticNetModel.evaluate(test_data)
test_results_ElasticNet.residuals.show()


# In[21]:


print("RSME: {}".format(test_results.rootMeanSquaredError))
print("R2: {}".format(test_results.r2))


# In[22]:


print("RSME(Lasso): {}".format(test_results_lasso.rootMeanSquaredError))
print("R2(Lasso): {}".format(test_results_lasso.r2))


# In[23]:


print("RSME(Ridge): {}".format(test_results_ridge.rootMeanSquaredError))
print("R2(Ridge): {}".format(test_results_ridge.r2))


# In[24]:


print("RSME(Elastic-Net): {}".format(test_results_ElasticNet.rootMeanSquaredError))
print("R2(Elastic-Net): {}".format(test_results_ElasticNet.r2))


# In[25]:


predictions_lr = lrModel.transform(test_data)
predictions_lr.select("prediction","log_rate","features").show()


# In[26]:


predictions_lasso = lr_lassoModel.transform(test_data)
predictions_lasso.select("prediction","log_rate","features").show()


# In[27]:


predictions_ridge = lr_ridgeModel.transform(test_data)
predictions_ridge.select("prediction","log_rate","features").show()


# In[28]:


predictions_ElasticNet = lr_ElasticNetModel.transform(test_data)
predictions_ElasticNet.select("prediction","log_rate","features").show()


# In[30]:


import numpy as np 
import matplotlib.pyplot as plt

beta=np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()


# In[33]:


beta_lasso=np.sort(lr_lassoModel.coefficients)
plt.plot(beta_lasso)
plt.ylabel('Beta Coefficients (Lasso)')
plt.show()


# In[32]:


beta_ridge=np.sort(lr_ridgeModel.coefficients)
plt.plot(beta_ridge)
plt.ylabel('Beta Coefficients (Ridge)')
plt.show()


# In[34]:


beta_ElasticNet=np.sort(lr_ElasticNetModel.coefficients)
plt.plot(beta_ElasticNet)
plt.ylabel('Beta Coefficients (Elastic-Net)')
plt.show()


# In[35]:


plt.scatter(x=predictions_lr.toPandas()["prediction"],y=predictions_lr.toPandas()["log_rate"])
plt.title('Make predictions of Linear Regression')
plt.xlabel('Predicted log_rate')
plt.ylabel('Actual log_rate')
plt.show()


# In[36]:


plt.scatter(x=predictions_lasso.toPandas()["prediction"],y=predictions_lasso.toPandas()["log_rate"])
plt.title('Make predictions of Lasso Regression')
plt.xlabel('Predicted log_rate')
plt.ylabel('Actual log_rate')
plt.show()


# In[37]:


plt.scatter(x=predictions_ridge.toPandas()["prediction"],y=predictions_ridge.toPandas()["log_rate"])
plt.title('Make predictions of Ridge Regression')
plt.xlabel('Predicted log_rate')
plt.ylabel('Actual log_rate')
plt.show()


# In[38]:


plt.scatter(x=predictions_ElasticNet.toPandas()["prediction"],y=predictions_ElasticNet.toPandas()["log_rate"])
plt.title('Make predictions of Elastic-Net')
plt.xlabel('Predicted log_rate')
plt.ylabel('Actual log_rate')
plt.show()

