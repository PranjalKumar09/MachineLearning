{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as pyplot\n",
    "import seaborn as sns\n",
    "from sklearn.impute import KNNImputer\n",
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
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
       "      <th>HomePlanet</th>\n",
       "      <th>CryoSleep</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Destination</th>\n",
       "      <th>Age</th>\n",
       "      <th>VIP</th>\n",
       "      <th>RoomService</th>\n",
       "      <th>FoodCourt</th>\n",
       "      <th>ShoppingMall</th>\n",
       "      <th>Spa</th>\n",
       "      <th>VRDeck</th>\n",
       "      <th>Name</th>\n",
       "      <th>Transported</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0001_01</td>\n",
       "      <td>Europa</td>\n",
       "      <td>False</td>\n",
       "      <td>B/0/P</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>39.0</td>\n",
       "      <td>False</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Maham Ofracculy</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0002_01</td>\n",
       "      <td>Earth</td>\n",
       "      <td>False</td>\n",
       "      <td>F/0/S</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>24.0</td>\n",
       "      <td>False</td>\n",
       "      <td>109.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>549.0</td>\n",
       "      <td>44.0</td>\n",
       "      <td>Juanna Vines</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0003_01</td>\n",
       "      <td>Europa</td>\n",
       "      <td>False</td>\n",
       "      <td>A/0/S</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>58.0</td>\n",
       "      <td>True</td>\n",
       "      <td>43.0</td>\n",
       "      <td>3576.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6715.0</td>\n",
       "      <td>49.0</td>\n",
       "      <td>Altark Susent</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0003_02</td>\n",
       "      <td>Europa</td>\n",
       "      <td>False</td>\n",
       "      <td>A/0/S</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>33.0</td>\n",
       "      <td>False</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1283.0</td>\n",
       "      <td>371.0</td>\n",
       "      <td>3329.0</td>\n",
       "      <td>193.0</td>\n",
       "      <td>Solam Susent</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0004_01</td>\n",
       "      <td>Earth</td>\n",
       "      <td>False</td>\n",
       "      <td>F/1/S</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>16.0</td>\n",
       "      <td>False</td>\n",
       "      <td>303.0</td>\n",
       "      <td>70.0</td>\n",
       "      <td>151.0</td>\n",
       "      <td>565.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Willy Santantines</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  PassengerId HomePlanet CryoSleep  Cabin  Destination   Age    VIP  \\\n",
       "0     0001_01     Europa     False  B/0/P  TRAPPIST-1e  39.0  False   \n",
       "1     0002_01      Earth     False  F/0/S  TRAPPIST-1e  24.0  False   \n",
       "2     0003_01     Europa     False  A/0/S  TRAPPIST-1e  58.0   True   \n",
       "3     0003_02     Europa     False  A/0/S  TRAPPIST-1e  33.0  False   \n",
       "4     0004_01      Earth     False  F/1/S  TRAPPIST-1e  16.0  False   \n",
       "\n",
       "   RoomService  FoodCourt  ShoppingMall     Spa  VRDeck               Name  \\\n",
       "0          0.0        0.0           0.0     0.0     0.0    Maham Ofracculy   \n",
       "1        109.0        9.0          25.0   549.0    44.0       Juanna Vines   \n",
       "2         43.0     3576.0           0.0  6715.0    49.0      Altark Susent   \n",
       "3          0.0     1283.0         371.0  3329.0   193.0       Solam Susent   \n",
       "4        303.0       70.0         151.0   565.0     2.0  Willy Santantines   \n",
       "\n",
       "   Transported  \n",
       "0        False  \n",
       "1         True  \n",
       "2        False  \n",
       "3        False  \n",
       "4         True  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_train = pd.read_csv('train.csv')\n",
    "df_test = pd.read_csv('test.csv')\n",
    "df_test['Transported'] = False\n",
    "df = pd.concat([df_train, df_test])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "HomePlanet      288\n",
       "CryoSleep       310\n",
       "Cabin           299\n",
       "Destination     274\n",
       "Age             270\n",
       "VIP             296\n",
       "RoomService     263\n",
       "FoodCourt       289\n",
       "ShoppingMall    306\n",
       "Spa             284\n",
       "VRDeck          268\n",
       "Name            294\n",
       "dtype: int64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum()[df.isna().sum()>0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)\n",
    "df  = df.drop(columns=['Name', 'Cabin'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df['Deck'] = df['Deck'].fillna('U')\n",
    "df['Num'] = df['Num'].fillna(-1)\n",
    "df['Side'] = df['Side'].fillna('U')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "encoder = LabelEncoder()\n",
    "df['Deck']  = encoder.fit_transform(df['Deck'])\n",
    "df['Side'] = encoder.fit_transform(df['Side'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
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
       "      <th>HomePlanet</th>\n",
       "      <th>CryoSleep</th>\n",
       "      <th>Destination</th>\n",
       "      <th>Age</th>\n",
       "      <th>VIP</th>\n",
       "      <th>RoomService</th>\n",
       "      <th>FoodCourt</th>\n",
       "      <th>ShoppingMall</th>\n",
       "      <th>Spa</th>\n",
       "      <th>VRDeck</th>\n",
       "      <th>Transported</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Num</th>\n",
       "      <th>Side</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0001_01</td>\n",
       "      <td>Europa</td>\n",
       "      <td>False</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>39.0</td>\n",
       "      <td>False</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>False</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0002_01</td>\n",
       "      <td>Earth</td>\n",
       "      <td>False</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>24.0</td>\n",
       "      <td>False</td>\n",
       "      <td>109.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>549.0</td>\n",
       "      <td>44.0</td>\n",
       "      <td>True</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0003_01</td>\n",
       "      <td>Europa</td>\n",
       "      <td>False</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>58.0</td>\n",
       "      <td>True</td>\n",
       "      <td>43.0</td>\n",
       "      <td>3576.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6715.0</td>\n",
       "      <td>49.0</td>\n",
       "      <td>False</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0003_02</td>\n",
       "      <td>Europa</td>\n",
       "      <td>False</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>33.0</td>\n",
       "      <td>False</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1283.0</td>\n",
       "      <td>371.0</td>\n",
       "      <td>3329.0</td>\n",
       "      <td>193.0</td>\n",
       "      <td>False</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0004_01</td>\n",
       "      <td>Earth</td>\n",
       "      <td>False</td>\n",
       "      <td>TRAPPIST-1e</td>\n",
       "      <td>16.0</td>\n",
       "      <td>False</td>\n",
       "      <td>303.0</td>\n",
       "      <td>70.0</td>\n",
       "      <td>151.0</td>\n",
       "      <td>565.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>True</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  PassengerId HomePlanet CryoSleep  Destination   Age    VIP  RoomService  \\\n",
       "0     0001_01     Europa     False  TRAPPIST-1e  39.0  False          0.0   \n",
       "1     0002_01      Earth     False  TRAPPIST-1e  24.0  False        109.0   \n",
       "2     0003_01     Europa     False  TRAPPIST-1e  58.0   True         43.0   \n",
       "3     0003_02     Europa     False  TRAPPIST-1e  33.0  False          0.0   \n",
       "4     0004_01      Earth     False  TRAPPIST-1e  16.0  False        303.0   \n",
       "\n",
       "   FoodCourt  ShoppingMall     Spa  VRDeck  Transported  Deck Num  Side  \n",
       "0        0.0           0.0     0.0     0.0        False     1   0     0  \n",
       "1        9.0          25.0   549.0    44.0         True     5   0     1  \n",
       "2     3576.0           0.0  6715.0    49.0        False     0   0     1  \n",
       "3     1283.0         371.0  3329.0   193.0        False     0   0     1  \n",
       "4       70.0         151.0   565.0     2.0         True     5   1     1  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "impute_lis = ['Age', 'VIP', 'Num', 'CryoSleep', 'Side', 'Deck', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']\n",
    "rest = list(set(df.columns)- set(impute_lis))\n",
    "df_rest = df[rest]\n",
    "imp = KNNImputer()\n",
    "df_imputed = imp.fit_transform(df[impute_lis])\n",
    "df_imputed = pd.DataFrame(df_imputed, columns = impute_lis )\n",
    "df = pd.concat([df_rest.reset_index(drop = True), df_imputed.reset_index(drop=True)], axis = 1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "bill_clos = ['RoomService', 'FoodCourt', 'ShoppingMall','Spa' ]\n",
    "df['mean_amt_spent'] = df[bill_clos].mean(axis =1)\n",
    "df['std_amt_spent'] = df[bill_clos].std(axis =1)\n",
    "\n",
    "df= df.fillna('U')\n",
    "\n",
    "category_colls  = ['Destination', 'HomePlanet']\n",
    "\n",
    "for col in category_colls:\n",
    "    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis = 1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop(columns = category_colls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Transported                  1.000000\n",
       "CryoSleep                    0.324373\n",
       "HomePlanet_Europa            0.131977\n",
       "Destination_55 Cancri e      0.083625\n",
       "Side                         0.067358\n",
       "FoodCourt                    0.034746\n",
       "PassengerId                  0.014628\n",
       "HomePlanet_U                 0.006403\n",
       "HomePlanet_Mars              0.005643\n",
       "ShoppingMall                 0.004154\n",
       "Destination_PSO J318.5-22    0.000760\n",
       "Destination_U               -0.000554\n",
       "VIP                         -0.018720\n",
       "Num                         -0.035240\n",
       "Age                         -0.050478\n",
       "Destination_TRAPPIST-1e     -0.072731\n",
       "std_amt_spent               -0.077729\n",
       "Deck                        -0.084981\n",
       "mean_amt_spent              -0.099098\n",
       "HomePlanet_Earth            -0.119644\n",
       "VRDeck                      -0.142783\n",
       "Spa                         -0.154832\n",
       "RoomService                 -0.174781\n",
       "Name: Transported, dtype: float64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.corr()['Transported'].sort_values(ascending = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Transported                  1.000000\n",
       "CryoSleep                    0.324373\n",
       "3_high_cols                  0.251587\n",
       "HomePlanet_Europa            0.131977\n",
       "Destination_55 Cancri e      0.083625\n",
       "Side                         0.067358\n",
       "FoodCourt                    0.034746\n",
       "PassengerId                  0.014628\n",
       "HomePlanet_U                 0.006403\n",
       "HomePlanet_Mars              0.005643\n",
       "ShoppingMall                 0.004154\n",
       "Destination_PSO J318.5-22    0.000760\n",
       "Destination_U               -0.000554\n",
       "VIP                         -0.018720\n",
       "Num                         -0.035240\n",
       "Age                         -0.050478\n",
       "Destination_TRAPPIST-1e     -0.072731\n",
       "std_amt_spent               -0.077729\n",
       "Deck                        -0.084981\n",
       "mean_amt_spent              -0.099098\n",
       "HomePlanet_Earth            -0.119644\n",
       "VRDeck                      -0.142783\n",
       "Spa                         -0.154832\n",
       "RoomService                 -0.174781\n",
       "2_low_cols                  -0.220362\n",
       "Name: Transported, dtype: float64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['3_high_cols'] = df['CryoSleep'] + df['HomePlanet_Europa'] + df['HomePlanet_Europa']+df['Destination_55 Cancri e']\n",
    "df['2_low_cols'] = df['RoomService'] + df['Spa'] + df['HomePlanet_Earth']\n",
    "df.corr()['Transported'].sort_values(ascending = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ! pip install lightbgm\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "# from lightbgm import LGBMClassifier\n",
    "\n",
    "df_train , df_test  = df[:df_train.shape[0]], df[df_train.shape[0]:]\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test = df_test.drop(columns=\"Transported\")\n",
    "df_train.shape, df_train.test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df_train.drop(columns=\"Transported\")\n",
    "Y = df_train[\"Transported\"]\n",
    "\n",
    "X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 23)\n",
    "\n",
    "model_1 = LogisticRegression()\n",
    "model_2 = DecisionTreeClassifier()\n",
    "model_3 = RandomForestClassifier()\n",
    "model_4 = XGBClassifier()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.78953421506613"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_1.fit(X_train, Y_train)\n",
    "pred = model_1.predict(X_test)\n",
    "accuracy_score(Y_test, pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7343300747556066"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_2.fit(X_train, Y_train)\n",
    "pred = model_2.predict(X_test)\n",
    "accuracy_score(Y_test, pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7981598619896493"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_3.fit(X_train, Y_train)\n",
    "pred = model_3.predict(X_test)\n",
    "accuracy_score(Y_test, pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:PassengerId: object",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[25], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[43mmodel_4\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX_train\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mY_train\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m      2\u001b[0m pred \u001b[38;5;241m=\u001b[39m model_4\u001b[38;5;241m.\u001b[39mpredict(X_test)\n\u001b[1;32m      3\u001b[0m accuracy_score(Y_test, pred)\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/core.py:729\u001b[0m, in \u001b[0;36mrequire_keyword_args.<locals>.throw_if.<locals>.inner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    727\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m k, arg \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mzip\u001b[39m(sig\u001b[38;5;241m.\u001b[39mparameters, args):\n\u001b[1;32m    728\u001b[0m     kwargs[k] \u001b[38;5;241m=\u001b[39m arg\n\u001b[0;32m--> 729\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/sklearn.py:1496\u001b[0m, in \u001b[0;36mXGBClassifier.fit\u001b[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)\u001b[0m\n\u001b[1;32m   1485\u001b[0m     params[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnum_class\u001b[39m\u001b[38;5;124m\"\u001b[39m] \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_classes_\n\u001b[1;32m   1487\u001b[0m (\n\u001b[1;32m   1488\u001b[0m     model,\n\u001b[1;32m   1489\u001b[0m     metric,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1494\u001b[0m     xgb_model, eval_metric, params, early_stopping_rounds, callbacks\n\u001b[1;32m   1495\u001b[0m )\n\u001b[0;32m-> 1496\u001b[0m train_dmatrix, evals \u001b[38;5;241m=\u001b[39m \u001b[43m_wrap_evaluation_matrices\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m   1497\u001b[0m \u001b[43m    \u001b[49m\u001b[43mmissing\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mmissing\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1498\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1499\u001b[0m \u001b[43m    \u001b[49m\u001b[43my\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1500\u001b[0m \u001b[43m    \u001b[49m\u001b[43mgroup\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m   1501\u001b[0m \u001b[43m    \u001b[49m\u001b[43mqid\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m   1502\u001b[0m \u001b[43m    \u001b[49m\u001b[43msample_weight\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msample_weight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1503\u001b[0m \u001b[43m    \u001b[49m\u001b[43mbase_margin\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbase_margin\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1504\u001b[0m \u001b[43m    \u001b[49m\u001b[43mfeature_weights\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfeature_weights\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1505\u001b[0m \u001b[43m    \u001b[49m\u001b[43meval_set\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43meval_set\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1506\u001b[0m \u001b[43m    \u001b[49m\u001b[43msample_weight_eval_set\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msample_weight_eval_set\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1507\u001b[0m \u001b[43m    \u001b[49m\u001b[43mbase_margin_eval_set\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbase_margin_eval_set\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1508\u001b[0m \u001b[43m    \u001b[49m\u001b[43meval_group\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m   1509\u001b[0m \u001b[43m    \u001b[49m\u001b[43meval_qid\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m   1510\u001b[0m \u001b[43m    \u001b[49m\u001b[43mcreate_dmatrix\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_create_dmatrix\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1511\u001b[0m \u001b[43m    \u001b[49m\u001b[43menable_categorical\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43menable_categorical\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1512\u001b[0m \u001b[43m    \u001b[49m\u001b[43mfeature_types\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfeature_types\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1513\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1515\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_Booster \u001b[38;5;241m=\u001b[39m train(\n\u001b[1;32m   1516\u001b[0m     params,\n\u001b[1;32m   1517\u001b[0m     train_dmatrix,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1526\u001b[0m     callbacks\u001b[38;5;241m=\u001b[39mcallbacks,\n\u001b[1;32m   1527\u001b[0m )\n\u001b[1;32m   1529\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28mcallable\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mobjective):\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/sklearn.py:534\u001b[0m, in \u001b[0;36m_wrap_evaluation_matrices\u001b[0;34m(missing, X, y, group, qid, sample_weight, base_margin, feature_weights, eval_set, sample_weight_eval_set, base_margin_eval_set, eval_group, eval_qid, create_dmatrix, enable_categorical, feature_types)\u001b[0m\n\u001b[1;32m    514\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_wrap_evaluation_matrices\u001b[39m(\n\u001b[1;32m    515\u001b[0m     missing: \u001b[38;5;28mfloat\u001b[39m,\n\u001b[1;32m    516\u001b[0m     X: Any,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    530\u001b[0m     feature_types: Optional[FeatureTypes],\n\u001b[1;32m    531\u001b[0m ) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Tuple[Any, List[Tuple[Any, \u001b[38;5;28mstr\u001b[39m]]]:\n\u001b[1;32m    532\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Convert array_like evaluation matrices into DMatrix.  Perform validation on the\u001b[39;00m\n\u001b[1;32m    533\u001b[0m \u001b[38;5;124;03m    way.\"\"\"\u001b[39;00m\n\u001b[0;32m--> 534\u001b[0m     train_dmatrix \u001b[38;5;241m=\u001b[39m \u001b[43mcreate_dmatrix\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    535\u001b[0m \u001b[43m        \u001b[49m\u001b[43mdata\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    536\u001b[0m \u001b[43m        \u001b[49m\u001b[43mlabel\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    537\u001b[0m \u001b[43m        \u001b[49m\u001b[43mgroup\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    538\u001b[0m \u001b[43m        \u001b[49m\u001b[43mqid\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mqid\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    539\u001b[0m \u001b[43m        \u001b[49m\u001b[43mweight\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msample_weight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    540\u001b[0m \u001b[43m        \u001b[49m\u001b[43mbase_margin\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbase_margin\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    541\u001b[0m \u001b[43m        \u001b[49m\u001b[43mfeature_weights\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfeature_weights\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    542\u001b[0m \u001b[43m        \u001b[49m\u001b[43mmissing\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mmissing\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    543\u001b[0m \u001b[43m        \u001b[49m\u001b[43menable_categorical\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43menable_categorical\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    544\u001b[0m \u001b[43m        \u001b[49m\u001b[43mfeature_types\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfeature_types\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    545\u001b[0m \u001b[43m        \u001b[49m\u001b[43mref\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m    546\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    548\u001b[0m     n_validation \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m eval_set \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(eval_set)\n\u001b[1;32m    550\u001b[0m     \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mvalidate_or_none\u001b[39m(meta: Optional[Sequence], name: \u001b[38;5;28mstr\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Sequence:\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/sklearn.py:954\u001b[0m, in \u001b[0;36mXGBModel._create_dmatrix\u001b[0;34m(self, ref, **kwargs)\u001b[0m\n\u001b[1;32m    952\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m _can_use_qdm(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtree_method) \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mbooster \u001b[38;5;241m!=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mgblinear\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[1;32m    953\u001b[0m     \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 954\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mQuantileDMatrix\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    955\u001b[0m \u001b[43m            \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mref\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mref\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnthread\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mn_jobs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmax_bin\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mmax_bin\u001b[49m\n\u001b[1;32m    956\u001b[0m \u001b[43m        \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    957\u001b[0m     \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m:  \u001b[38;5;66;03m# `QuantileDMatrix` supports lesser types than DMatrix\u001b[39;00m\n\u001b[1;32m    958\u001b[0m         \u001b[38;5;28;01mpass\u001b[39;00m\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/core.py:729\u001b[0m, in \u001b[0;36mrequire_keyword_args.<locals>.throw_if.<locals>.inner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    727\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m k, arg \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mzip\u001b[39m(sig\u001b[38;5;241m.\u001b[39mparameters, args):\n\u001b[1;32m    728\u001b[0m     kwargs[k] \u001b[38;5;241m=\u001b[39m arg\n\u001b[0;32m--> 729\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/core.py:1528\u001b[0m, in \u001b[0;36mQuantileDMatrix.__init__\u001b[0;34m(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, max_bin, ref, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)\u001b[0m\n\u001b[1;32m   1508\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28many\u001b[39m(\n\u001b[1;32m   1509\u001b[0m         info \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m   1510\u001b[0m         \u001b[38;5;28;01mfor\u001b[39;00m info \u001b[38;5;129;01min\u001b[39;00m (\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1521\u001b[0m         )\n\u001b[1;32m   1522\u001b[0m     ):\n\u001b[1;32m   1523\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[1;32m   1524\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mIf data iterator is used as input, data like label should be \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m   1525\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mspecified as batch argument.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m   1526\u001b[0m         )\n\u001b[0;32m-> 1528\u001b[0m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_init\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m   1529\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdata\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1530\u001b[0m \u001b[43m    \u001b[49m\u001b[43mref\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mref\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1531\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlabel\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mlabel\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1532\u001b[0m \u001b[43m    \u001b[49m\u001b[43mweight\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mweight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1533\u001b[0m \u001b[43m    \u001b[49m\u001b[43mbase_margin\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbase_margin\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1534\u001b[0m \u001b[43m    \u001b[49m\u001b[43mgroup\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1535\u001b[0m \u001b[43m    \u001b[49m\u001b[43mqid\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mqid\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1536\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlabel_lower_bound\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mlabel_lower_bound\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1537\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlabel_upper_bound\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mlabel_upper_bound\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1538\u001b[0m \u001b[43m    \u001b[49m\u001b[43mfeature_weights\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfeature_weights\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1539\u001b[0m \u001b[43m    \u001b[49m\u001b[43mfeature_names\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfeature_names\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1540\u001b[0m \u001b[43m    \u001b[49m\u001b[43mfeature_types\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfeature_types\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1541\u001b[0m \u001b[43m    \u001b[49m\u001b[43menable_categorical\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43menable_categorical\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1542\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/core.py:1587\u001b[0m, in \u001b[0;36mQuantileDMatrix._init\u001b[0;34m(self, data, ref, enable_categorical, **meta)\u001b[0m\n\u001b[1;32m   1575\u001b[0m config \u001b[38;5;241m=\u001b[39m make_jcargs(\n\u001b[1;32m   1576\u001b[0m     nthread\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mnthread, missing\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmissing, max_bin\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmax_bin\n\u001b[1;32m   1577\u001b[0m )\n\u001b[1;32m   1578\u001b[0m ret \u001b[38;5;241m=\u001b[39m _LIB\u001b[38;5;241m.\u001b[39mXGQuantileDMatrixCreateFromCallback(\n\u001b[1;32m   1579\u001b[0m     \u001b[38;5;28;01mNone\u001b[39;00m,\n\u001b[1;32m   1580\u001b[0m     it\u001b[38;5;241m.\u001b[39mproxy\u001b[38;5;241m.\u001b[39mhandle,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1585\u001b[0m     ctypes\u001b[38;5;241m.\u001b[39mbyref(handle),\n\u001b[1;32m   1586\u001b[0m )\n\u001b[0;32m-> 1587\u001b[0m \u001b[43mit\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mreraise\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1588\u001b[0m \u001b[38;5;66;03m# delay check_call to throw intermediate exception first\u001b[39;00m\n\u001b[1;32m   1589\u001b[0m _check_call(ret)\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/core.py:575\u001b[0m, in \u001b[0;36mDataIter.reraise\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    573\u001b[0m exc \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_exception\n\u001b[1;32m    574\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_exception \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m--> 575\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m exc\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/core.py:556\u001b[0m, in \u001b[0;36mDataIter._handle_exception\u001b[0;34m(self, fn, dft_ret)\u001b[0m\n\u001b[1;32m    553\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m dft_ret\n\u001b[1;32m    555\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 556\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfn\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    557\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:  \u001b[38;5;66;03m# pylint: disable=broad-except\u001b[39;00m\n\u001b[1;32m    558\u001b[0m     \u001b[38;5;66;03m# Defer the exception in order to return 0 and stop the iteration.\u001b[39;00m\n\u001b[1;32m    559\u001b[0m     \u001b[38;5;66;03m# Exception inside a ctype callback function has no effect except\u001b[39;00m\n\u001b[1;32m    560\u001b[0m     \u001b[38;5;66;03m# for printing to stderr (doesn't stop the execution).\u001b[39;00m\n\u001b[1;32m    561\u001b[0m     tb \u001b[38;5;241m=\u001b[39m sys\u001b[38;5;241m.\u001b[39mexc_info()[\u001b[38;5;241m2\u001b[39m]\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/core.py:640\u001b[0m, in \u001b[0;36mDataIter._next_wrapper.<locals>.<lambda>\u001b[0;34m()\u001b[0m\n\u001b[1;32m    637\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_data_ref \u001b[38;5;241m=\u001b[39m ref\n\u001b[1;32m    639\u001b[0m \u001b[38;5;66;03m# pylint: disable=not-callable\u001b[39;00m\n\u001b[0;32m--> 640\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_handle_exception(\u001b[38;5;28;01mlambda\u001b[39;00m: \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mnext\u001b[49m\u001b[43m(\u001b[49m\u001b[43minput_data\u001b[49m\u001b[43m)\u001b[49m, \u001b[38;5;241m0\u001b[39m)\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/data.py:1280\u001b[0m, in \u001b[0;36mSingleBatchInternalIter.next\u001b[0;34m(self, input_data)\u001b[0m\n\u001b[1;32m   1278\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;241m0\u001b[39m\n\u001b[1;32m   1279\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mit \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;241m1\u001b[39m\n\u001b[0;32m-> 1280\u001b[0m \u001b[43minput_data\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1281\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;241m1\u001b[39m\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/core.py:729\u001b[0m, in \u001b[0;36mrequire_keyword_args.<locals>.throw_if.<locals>.inner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    727\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m k, arg \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mzip\u001b[39m(sig\u001b[38;5;241m.\u001b[39mparameters, args):\n\u001b[1;32m    728\u001b[0m     kwargs[k] \u001b[38;5;241m=\u001b[39m arg\n\u001b[0;32m--> 729\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/core.py:623\u001b[0m, in \u001b[0;36mDataIter._next_wrapper.<locals>.input_data\u001b[0;34m(data, feature_names, feature_types, **kwargs)\u001b[0m\n\u001b[1;32m    621\u001b[0m     new, cat_codes, feature_names, feature_types \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_temporary_data\n\u001b[1;32m    622\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m--> 623\u001b[0m     new, cat_codes, feature_names, feature_types \u001b[38;5;241m=\u001b[39m \u001b[43m_proxy_transform\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    624\u001b[0m \u001b[43m        \u001b[49m\u001b[43mdata\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    625\u001b[0m \u001b[43m        \u001b[49m\u001b[43mfeature_names\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    626\u001b[0m \u001b[43m        \u001b[49m\u001b[43mfeature_types\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    627\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_enable_categorical\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    628\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    629\u001b[0m \u001b[38;5;66;03m# Stage the data, meta info are copied inside C++ MetaInfo.\u001b[39;00m\n\u001b[1;32m    630\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_temporary_data \u001b[38;5;241m=\u001b[39m (new, cat_codes, feature_names, feature_types)\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/data.py:1315\u001b[0m, in \u001b[0;36m_proxy_transform\u001b[0;34m(data, feature_names, feature_types, enable_categorical)\u001b[0m\n\u001b[1;32m   1313\u001b[0m     data \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mDataFrame(data)\n\u001b[1;32m   1314\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m _is_pandas_df(data):\n\u001b[0;32m-> 1315\u001b[0m     arr, feature_names, feature_types \u001b[38;5;241m=\u001b[39m \u001b[43m_transform_pandas_df\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m   1316\u001b[0m \u001b[43m        \u001b[49m\u001b[43mdata\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43menable_categorical\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mfeature_names\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mfeature_types\u001b[49m\n\u001b[1;32m   1317\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1318\u001b[0m     arr, _ \u001b[38;5;241m=\u001b[39m _ensure_np_dtype(arr, arr\u001b[38;5;241m.\u001b[39mdtype)\n\u001b[1;32m   1319\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m arr, \u001b[38;5;28;01mNone\u001b[39;00m, feature_names, feature_types\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/data.py:490\u001b[0m, in \u001b[0;36m_transform_pandas_df\u001b[0;34m(data, enable_categorical, feature_names, feature_types, meta, meta_type)\u001b[0m\n\u001b[1;32m    483\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m dtype \u001b[38;5;129;01min\u001b[39;00m data\u001b[38;5;241m.\u001b[39mdtypes:\n\u001b[1;32m    484\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\n\u001b[1;32m    485\u001b[0m         (dtype\u001b[38;5;241m.\u001b[39mname \u001b[38;5;129;01min\u001b[39;00m _pandas_dtype_mapper)\n\u001b[1;32m    486\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m is_pd_sparse_dtype(dtype)\n\u001b[1;32m    487\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m (is_pd_cat_dtype(dtype) \u001b[38;5;129;01mand\u001b[39;00m enable_categorical)\n\u001b[1;32m    488\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m is_pa_ext_dtype(dtype)\n\u001b[1;32m    489\u001b[0m     ):\n\u001b[0;32m--> 490\u001b[0m         \u001b[43m_invalid_dataframe_dtype\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdata\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    491\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m is_pa_ext_dtype(dtype):\n\u001b[1;32m    492\u001b[0m         pyarrow_extension \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m\n",
      "File \u001b[0;32m/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/xgboost/data.py:308\u001b[0m, in \u001b[0;36m_invalid_dataframe_dtype\u001b[0;34m(data)\u001b[0m\n\u001b[1;32m    306\u001b[0m type_err \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mDataFrame.dtypes for data must be int, float, bool or category.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    307\u001b[0m msg \u001b[38;5;241m=\u001b[39m \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\"\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mtype_err\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m_ENABLE_CAT_ERR\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m \u001b[39m\u001b[38;5;132;01m{\u001b[39;00merr\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\"\"\u001b[39m\n\u001b[0;32m--> 308\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(msg)\n",
      "\u001b[0;31mValueError\u001b[0m: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:PassengerId: object"
     ]
    }
   ],
   "source": [
    "model_4.fit(X_train, Y_train)\n",
    "pred = model_4.predict(X_test)\n",
    "accuracy_score(Y_test, pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred = model_3.predict(df_test)\n",
    "final = pd.DataFrame()\n",
    "final['PassengerId'] = df_test['PassengerId']\n",
    "final['Transported'] = pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "final.to_csv('prediction.csv', index = False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "cloudspace",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
