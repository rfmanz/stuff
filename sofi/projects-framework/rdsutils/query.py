"""
query.py

Query helper functions.
"""

import pandas as pd

def query_postgres(query, port, database='postgres'):
    """Query a postgres database using a string query and port.
    
    Parameters
    ----------
    query : string
        Query to be exectued.
    port : int/string
        Port through which the database is connected.
        
    Returns
    -------
    df : pandas DataFrame
    """
    from sqlalchemy import create_engine
    engine = create_engine("postgres://127.0.0.1:{}/{}".format(str(port),
                                                               database))
    _ = engine.connect()
    
    return pd.read_sql(query, engine)


# def query_mysql(query, port, database='sofi'):
#     """Query a mysql database using a string query and port.
    
#     Parameters
#     ----------
#     query : string
#         Query to be exectued.
#     port : int/string
#         Port through which the database is connected.
        
#     Returns
#     -------
#     df : pandas DataFrame
#     """
#     from MySQLdb import connect
#     conn = connect(**{"user":"",
#                       "passwd":"",
#                       "host":"127.0.0.1",
#                       "port":port,
#                       "db":database})                      
    
#     return pd.read_sql(query, con=conn)
