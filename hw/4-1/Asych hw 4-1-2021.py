#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#older versions of python will need to bypass certification error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#Create your own column headers using a python list
col_names = ['Id',
             'Survived',
             'Passenger Class',
             'Full Name',
             'Gender',
             'Age',
             'Sibling Spouse Count',
             'Parent Children Count',
             'Ticket Number',
             'Price', 'Cabin',
             'Station']

# read in the dataset as a pandas DataFrame, calling it df
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv', names=col_names, skiprows=[0])
# alternatively:
# df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
# df.columns = col_names

#Print the whole dataframe
print(df)

#Print first 5 Rows
#print("First 5 rows:")
#print(df.head(5))

#Adding constant to Age (make everyone 10 years older)
df['Age'] = df['Age'] + 10

#Adding 2 numerical columns to get a new one
df['Family Count'] = df['Sibling Spouse Count'] + df['Parent Children Count']

#Delete columns
df=df.drop(columns=['Passenger Class', 'Gender', 'Sibling Spouse Count','Parent Children Count','Ticket Number','Price','Cabin','Station'])
# alternatively: df=df.drop(['Passenger Class','Gender','Sibling Spouse Count','Parent Children Count','Ticket Number','Price','Cabin','Station'], axis=1)

#if statement on a column --print data for anyone less than 20 years old (hint: print out (df['Age'] <=20), it's called a boolean list, can you figure out what it represents?)
p = df[(df['Age'] <= 20)]

#Sorting the whole data using a specific column. Note we're not storing the sorted p back into p, merely printing it, so p remains unchanged
print(p.sort_values(by=['Age', 'Survived']))


# In[8]:


#Question 10
df2 = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv', names=['Id','Survived','Passenger Class','Full Name','Gender','Age','Sibling Spouse Count','Parent Children Count','Ticket Number','Price', 'Cabin','Station'], skiprows=[0])
print(df2[df2['Survived']==1].sort_values(by=['Gender', 'Station','Age']))


# In[66]:
#converts a date into the year, represented by an integer for comparison
def dateConvert(dat):
    return int(dat.split()[2])

#Question 11
df11=pd.read_csv("netflix_titles.csv")
print(df11.size)
#removes values with null, note that we only remove from the columns type,date_added, as these are the columns we use, and if we just run a dropna, we lose nearly half of our data
df11.dropna(subset = ['type','date_added'], inplace=True)
#check how many values were deleted
print(df11.size)
#convert values
df11['year_added']=df11['date_added'].apply(dateConvert)
#Split between TVshows and movies
TVshow= df11[df11['type']=='TV Show']
Movie = df11[df11['type']=='Movie']
#Not sure where this goes wrong, I may have approached the problem incorrectly, but this is the data I found by taking the content added before 2010, then between 2010->present
print("Pre-2010, there were "+str(TVshow[TVshow['year_added']<=2010].size)+" TV shows. From 2010-2020, "+str(TVshow[TVshow['year_added']>2010].size)+" shows were added.")
print("Pre-2010, there were "+str(Movie[Movie['year_added']<=2010].size)+" Movies. From 2010-2020, "+str(Movie[Movie['year_added']>2010].size)+" shows were added.")


# In[53]:




