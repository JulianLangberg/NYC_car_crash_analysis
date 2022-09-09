from databricks import sql
import pandas as pd

def tabla(db,tabla):
    connection =  sql.connect(server_hostname = 'https://dbc-bd9b117a-06cc.cloud.databricks.com',
                              http_path       = 'sql/protocolv1/o/4317428421786124/0908-202625-7ocde901',
                              access_token    = 'dapi025ca0bebe2941d57eee4ccbd170cace')

    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM {db}.{tabla}")
    result = cursor.fetchall()
    df = pd.DataFrame(result)
    cursor.close()
    connection.close()
    return df