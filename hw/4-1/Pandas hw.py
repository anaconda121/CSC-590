#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""""
1. The output produced are the dataframe (df), with the values sorted by age (integer) and survived (boolean). 
The reason the other columns that are printed out in df.head() earlier in the code are not printed out at the end is 
because of the df.drop(....).

2. The info appears to be a dataset about passengers aboard the titanic. It contains values such as their name, 
whether they survied or not, their class aboard the ship among others.

3. The data is coming from the url (https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv). 
It is also available in a formatted version @ https://github.com/datasciencedojo/datasets/blob/master/titanic.csv.

4. The data is in csv format, where each value in a row is separated by columns. The first row in the dataset is 
contains the names of all the columns in the dataset.

5. The data is loaded in using this line of code: df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv', names=col_names, skiprows=[0]).
What this does it store the output of the pd.read_csv() function in a Pandas dataframe named df. The first parameter is the url of the CSV file. 
The second parameter loads in the names of the columns for the dataset by referencing a col_names array containing all the column
names in strings. The third parameter tells the function what row(s) to skip when reading in the data. The [0] tells the function
to skip the first row, which contains the column names that the creator of the dataset choose. But we wanted to rename these columns
to be more descriptive, which explains why we skipped the first row and instead inserted an array of our own column names.

6. This is code is able to edit all the values of 1 column simutaneously, delete unneeded columns, filter the dataset
based on certain parameters, and sort a dataset based on certain parameters. 

7. The variable df has a type of pandas dataframe, and stores information such as strings for names of passengers, integers for age, passenger class, and passenger id.

8. The array col_names contains the new names that we would have the dataset to have. It is called when we load in the data
using pd.read_csv(...), with the paramter names = col_names. This will load in the column names as those in the col_names array.

9. If we print(df['Age'] <= 20), we get a pandas boolean series of size 891. For each row in the dataset, it stores true if 
the age of that passenger is <= 20, and false if not. The line p = df[(df['Age'] <= 20)] uses that boolean series to determine
what rows of df to put into p. If df['Age'] <= 20 at a certain row, that row will be added into p, otherwise it will not. 
The second line sorts p by least age. If the ages of mutliple passengers is the same, it will use the value in the 'Survived' 
column as a tiebreaker, where 0 edges out 1.

10. 

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv', names=col_names, skiprows=[0])

# true = 1 , false = 0
survived = df[(df['Survived'] == 1)]

survived.sort_values(by = ['Gender', 'Station', 'Age'])
survived

11.

### QUESTION 11

netflix = pd.read_csv('netflix_titles.csv')

netflix = netflix.drop(columns =['show_id', 'title', 'director', 'cast', 'country', 'release_year', 'rating', 'duration', 'listed_in', 'description'])
netflix.dropna(subset = ['type','date_added'], inplace=True)

netflix['year_added'] = netflix['date_added'].apply(lambda x : int(x.split(", ")[1]))

tv_shows_before_2010 = netflix[(netflix['type'] == 'TV Show') & (netflix['year_added'] <= 2010)]
tv_shows_after_2010 = netflix[(netflix['type'] == 'TV Show') & (netflix['year_added'] > 2010)]

movies_before_2010 = netflix[(netflix['type'] == 'Movie') & (netflix['year_added'] <= 2010)]
movies_after_2010 = netflix[(netflix['type'] == 'Movie') & (netflix['year_added'] > 2010)]

print("BEFORE 2010:")
print("TV-SHOWS:", len(tv_shows_before_2010.index))
print("MOVIES:", len(movies_before_2010.index))

print("\nAFTER 2010:")
print("TV-SHOWS:", len(tv_shows_after_2010.index))
print("MOVIES:", len(movies_after_2010.index))
""""


# In[8]:


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
print(type(df))
# alternatively:
# df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
# df.columns = col_names

#Print the whole dataframe
#print(df)

#Print first 5 Rows
print("First 5 rows:")
print(df.head(5))

#Adding constant to Age (make everyone 10 years older)
df['Age'] = df['Age'] + 10

#Adding 2 numerical columns to get a new one
df['Family Count'] = df['Sibling Spouse Count'] + df['Parent Children Count']

#Delete columns
df=df.drop(columns=['Passenger Class', 'Gender', 'Sibling Spouse Count','Parent Children Count','Ticket Number','Price','Cabin','Station'])
# alternatively: df=df.drop(['Passenger Class','Gender','Sibling Spouse Count','Parent Children Count','Ticket Number','Price','Cabin','Station'], axis=1)

#if statement on a column --print data for anyone less than 20 years old (hint: print out (df['Age'] <=20), it's called a boolean list, can you figure out what it represents?)
print(df['Age'] <= 20)
print(type(df['Age'] <= 20))
p = df[(df['Age'] <= 20)]

#Sorting the whole data using a specific column. Note we're not storing the sorted p back into p, merely printing it, so p remains unchanged
print("\n\nFiltered")
print(p.sort_values(by=['Age', 'Survived']))


# In[13]:


### Question 10

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv', names=col_names, skiprows=[0])

# true = 1 , false = 0
survived = df[(df['Survived'] == 1)]

survived.sort_values(by = ['Gender', 'Station', 'Age'])
survived


# In[88]:



netflix


# In[121]:


### QUESTION 11

netflix = pd.read_csv('netflix_titles.csv')

netflix = netflix.drop(columns =['show_id', 'title', 'director', 'cast', 'country', 'release_year', 'rating', 'duration', 'listed_in', 'description'])
netflix.dropna(subset = ['type','date_added'], inplace=True)

netflix['year_added'] = netflix['date_added'].apply(lambda x : int(x.split(", ")[1]))

tv_shows_before_2010 = netflix[(netflix['type'] == 'TV Show') & (netflix['year_added'] <= 2010)]
tv_shows_after_2010 = netflix[(netflix['type'] == 'TV Show') & (netflix['year_added'] > 2010)]

movies_before_2010 = netflix[(netflix['type'] == 'Movie') & (netflix['year_added'] <= 2010)]
movies_after_2010 = netflix[(netflix['type'] == 'Movie') & (netflix['year_added'] > 2010)]

print("BEFORE 2010:")
print("TV-SHOWS:", len(tv_shows_before_2010.index))
print("MOVIES:", len(movies_before_2010.index))

print("\nAFTER 2010:")
print("TV-SHOWS:", len(tv_shows_after_2010.index))
print("MOVIES:", len(movies_after_2010.index))


# In[119]:


5373+2399+4+1


# In[122]:


netflix


# In[21]:


movies_before_2010


# In[22]:


movies_after_2010


# In[23]:


tv_shows_before_2010


# In[24]:


tv_shows_after_2010


# In[ ]:




