# Running Optuna

These are the steps needed to run Optuna.

## Use Built-in mysqlite3 with Optuna Dashboard

Will just be for a single instance -- one User.  Use Optuna Dashboard.  Be sure to install `pip install opoutna-dashboard` having already installed optuna.

```.bash
optuna-dashboard sqlite:///db.sqlite3
```

## Running optuna with Mysql Database

That will setup a Mysql Server -- a container -- that can server multiple instances.  

```.bash

docker run --name optuna-mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -e MYSQL_DATABASE=optuna_db -e MYSQL_USER=optuna_user -e MYSQL_PASSWORD=optuna_pw -p 3306:3306 -d mysql:latest

```

More to follow.
