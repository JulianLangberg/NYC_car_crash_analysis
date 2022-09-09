from databricks import sql
import pandas as pd

def tabla(db,tabla):
    connection =  sql.connect(server_hostname = 'dbc-bd9b117a-06cc.cloud.databricks.com',
                              http_path       = 'sql/protocolv1/o/4317428421786124/0908-182202-5nayurek',
                              access_token    = 'dapi353b2e985693bece2b9d640481f58fc4')

    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM {db}.{tabla}")
    result = cursor.fetchall()
    columns = cursor.description
    field_names = [i[0] for i in columns]
    df = pd.DataFrame(result, columns=field_names)
    cursor.close()
    connection.close()
    return df 
    