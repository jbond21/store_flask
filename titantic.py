{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled1.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyNoFiJajKq0lFRlPku38Usp",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/jbond21/store_flask/blob/master/titantic.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bJCojlKMH9ey",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n"
      ],
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "38d6f2uKINm8",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "df = pd.read_csv('train.csv')"
      ],
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Jn9eScLgIeEi",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "450fe1a3-b76e-4aa3-cd75-5adc42fd9571"
      },
      "source": [
        "print(df.shape)"
      ],
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(891, 12)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CCpSHc_9L2nP",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 496
        },
        "outputId": "c64e0c41-fb5c-4e35-afd5-b772896bff03"
      },
      "source": [
        "df.head(5)"
      ],
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked\n",
              "0            1         0       3  ...   7.2500   NaN         S\n",
              "1            2         1       1  ...  71.2833   C85         C\n",
              "2            3         1       3  ...   7.9250   NaN         S\n",
              "3            4         1       1  ...  53.1000  C123         S\n",
              "4            5         0       3  ...   8.0500   NaN         S\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r5HUS8EkMCRP",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "df_test = pd.read_csv('test.csv')"
      ],
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "b_SJOXFYNQr7",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 71
        },
        "outputId": "be4fad06-1f71-4a76-d39d-2df846cdecdd"
      },
      "source": [
        "def get_title(name):\n",
        "    if '.' in name:\n",
        "        return name.split(',')[1].split('.')[0].strip()\n",
        "    else:\n",
        "        return 'title not in name'\n",
        "    \n",
        "titles = sorted(set([x for x in df.Name.map(lambda x: get_title(x))]))\n",
        "print(titles)\n",
        "print(len(titles), ':', titles )"
      ],
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir', 'the Countess']\n",
            "17 : ['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir', 'the Countess']\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Xg2nauRLQAmP",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def shorter_title(x):\n",
        "    title = x['Title']\n",
        "    if title in ['Capt', 'Col', 'Major']:\n",
        "        return 'Officer'\n",
        "    elif title in ['the Countess', 'Don', 'Sir', 'Jonkheer']:\n",
        "        return 'Royalty'\n",
        "    elif title in  ['Mme', 'Lady']:\n",
        "        return 'Mrs'\n",
        "    elif title in ['Mlle', 'Ms']:\n",
        "        return 'Miss'\n",
        "    else:\n",
        "        return title"
      ],
      "execution_count": 42,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AdtdR1Z4Q7F2",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "df['Title']= df['Name'].map(lambda x: get_title(x)) \n"
      ],
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CG6_yie1RHi-",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 170
        },
        "outputId": "1bbb3acb-75ec-43fa-8853-d2b5997d3e13"
      },
      "source": [
        "df['Title']= df.apply(shorter_title, axis=1)\n",
        "print(df.Title.value_counts())"
      ],
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mr         517\n",
            "Miss       185\n",
            "Mrs        127\n",
            "Master      40\n",
            "Dr           7\n",
            "Rev          6\n",
            "Officer      5\n",
            "Royalty      4\n",
            "Name: Title, dtype: int64\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rPrzU8_PUnCa",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 459
        },
        "outputId": "e18e3679-2000-4ba7-efd4-1b195c04723c"
      },
      "source": [
        "df['Age'].fillna(df['Age'].median(), inplace=True)\n",
        "df['Fare'].fillna(df['Fare'].median(), inplace=True)\n",
        "df['Embarked'].fillna('S', inplace=True)\n",
        "df.drop('Cabin', axis=1, inplace=True)\n",
        "df.drop('Ticket', axis=1, inplace=True)\n",
        "df.drop('Name', axis=1, inplace=True)\n",
        "df.Sex.replace(('male', 'female'), (0,1), inplace=True)\n",
        "df.Embarked.replace(('S', 'C', 'Q'), (0,1,2), inplace=True)\n",
        "df.Title.replace(('Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Officer', 'Royalty'), (0,1,2,3,4,5,6,7), inplace=True)\n",
        "print(df.isnull().sum())\n",
        "print(df['Sex'].sample(5))\n",
        "print(df['Embarked'].sample(5))\n",
        "print(df.columns)"
      ],
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "PassengerId    0\n",
            "Survived       0\n",
            "Pclass         0\n",
            "Sex            0\n",
            "Age            0\n",
            "SibSp          0\n",
            "Parch          0\n",
            "Fare           0\n",
            "Embarked       0\n",
            "Title          0\n",
            "dtype: int64\n",
            "610    1\n",
            "174    0\n",
            "574    0\n",
            "627    1\n",
            "741    0\n",
            "Name: Sex, dtype: int64\n",
            "68     0\n",
            "781    0\n",
            "397    0\n",
            "186    2\n",
            "426    0\n",
            "Name: Embarked, dtype: int64\n",
            "Index(['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',\n",
            "       'Fare', 'Embarked', 'Title'],\n",
            "      dtype='object')\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "O2kqzN8PZwaQ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "x = df.drop(['Survived', 'PassengerId'], axis=1)\n",
        "y = df['Survived']\n",
        "x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1)"
      ],
      "execution_count": 46,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8iSvTdxIaf6q",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "4a707c79-f839-4f40-e847-4f396b0f62bd"
      },
      "source": [
        "import pickle\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "randomforest = RandomForestClassifier()\n",
        "randomforest.fit(x_train, y_train)\n",
        "y_pred= randomforest.predict(x_val)\n",
        "acc_randomforest = round(accuracy_score(y_pred, y_val)*100,2)\n",
        "print(acc_randomforest)\n",
        "\n",
        "pickle.dump(randomforest, open('titantic_model.sav', 'wb'))"
      ],
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "84.44\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}