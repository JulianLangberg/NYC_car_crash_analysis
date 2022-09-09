
from streamlit.proto import Image_pb2
# Libraries
import streamlit as st 
import pandas as pd
from PIL import Image
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter
import pydeck as pdk

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

## Carga DF
from conection_databricks import tabla


header1 = st.container()
#dataset = st.container()
mod1 = st.container()


# Empezamos con CSV.
#@st.cache(allow_output_mutation=True)
#def load_data():
#file = '/content/drive/MyDrive/Colab Notebooks/Motor_Vehicle_Collisions-Crashes1.csv'
#file = 'Motor_Vehicle_Collisions-Crashes1.csv'
#df3 = pd.read_csv(file)
#  return df3
df3 = tabla(gold,df_definitivo)
df_semana_siniestros = tabla(gold,df_semana_siniestros)
df_motivos_grouped_aux_pre = tabla(gold,df_motivos_grouped_aux_pre)
df_motivos_grouped_aux_post =  tabla(gold,df_motivos_grouped_aux_post)
df_motivos_grouped_aux_post_viernes = tabla(gold,df_motivos_grouped_aux_post_viernes)
df_motivos_grouped_post_winter = tabla(gold,df_motivos_grouped_post_winter)
#df3 = load_data()

df_definitivo = df3.to_pandas()
df = df3
## cargar df_definitivo
## df_season_siniestros
## df_season_lockdown
## df_semana_siniestros
## df_week_lockdown

###########  VISUAL, PRESENTACION EQUIPO

with header1:
    st.title("Siniestralidad vial en NYC: análisis descriptivo y su impacto en la economía")
    image = Image.open('/content/drive/MyDrive/Colab Notebooks/images/portada_nyc_car_crash.jpg')

    st.image(image, caption='NYC - Car crash') 


    st.markdown("En la presente investigación se utilizará información oficial sobre siniestralidad vial recolectada desde https://opendata.cityofnewyork.us/, información climatológica del National Centers for Environmental Information en https://www.ncei.noaa.gov/, y los costos económicos generados surgen de investigaciones presentadas por el National Safety Counsil en https://injuryfacts.nsc.org/ ")
    st.markdown("Equipo de trabajo:")
    st.markdown("Arnone, Miguel")
    st.markdown("Langberg, Julian")
    st.markdown("Ojeda, Guillermo Agustín")
    st.markdown("Villarraga-Morales, Carolina")
with mod1:
    st.title("A total NYC, podemos apreciar una fuerte caída en la siniestralidad en los últimos dos año ")
    st.markdown("En el gráfico se observa como la cantidad de siniestros se reducen posterior al lockdown en el comienzo de la pandemia por Covid-19")

###### TRAIGO TABLAS, ETL  Y  GRAFICO INTRODUCTORIO 



#### Grafico evolutivo

fig = px.line(df_definitivo, x = df_definitivo['crash_date'],
                y=df_definitivo['siniestros_media_movil'], 
                title='Evolución siniestros NYC', width=800, height=550,
                labels= {'crash_date':'Fechas','value':'Cantidad'})

fig.add_trace(
    go.Scatter(x=list(df_definitivo.crash_date), y=list(df_definitivo['siniestros_media_movil'])
                , name='Cantidad de siniestros_media_móvil'))
# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

st.plotly_chart(fig, use_container_width=True)

## Grafico evolutivo

st.markdown("Haciendo foco en la cantidad de heridos, podemos encontrar un patrón de máximos y mínimos")

fig = px.line(df_definitivo,x = df_definitivo['crash_date'],
                y=['Lastimados_media_movil','Muertos_media_movil'], 
                title='Evolución siniestros NYC', width=800, height=550,
                labels= {'crash_date':'Fechas','value':'Cantidad'})

fig.add_trace(
    go.Scatter(x=list(df_definitivo.crash_date), y=list(df_definitivo['Lastimados_media_movil'])
                , name='Cantidad de personas lastimadas'))
# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

st.plotly_chart(fig, use_container_width=True)


#st.title("A total NYC, podemos apreciar una fuerte caída en la siniestralidad en los últimos dos año ")
st.markdown("Al profundizar en estos valores por temporada, observamos que los picos de heridos por siniestros de tránsito ocurren en verano")

#### ETL PARA GRAFICO DE TEMPORADASS

df_season_siniestros= df_definitivo.groupby('season')[['number_of_crashes']].sum().reset_index()


fig = px.bar(df_season_siniestros, x="season", y=["number_of_crashes"], title="Siniestros por temporada")

st.plotly_chart(fig, use_container_width=True)


################
#### ETL PARA GRAFICO DE COVID

#df_season_lockdown=df_definitivo[['lockdown','season','number_of_crashes']]

#ec = ['blue', 'green', 'orange', 'grey']

#df_season_lockdown.groupby(['season','lockdown']).mean().unstack('season').plot.bar(figsize=(10,4),color=ec)


################# ver titulooo



st.markdown("De similar forma, se observa que los días viernes también existía mayor cantidad de siniestros")



#### ETL PARA GRAFICO DE DIAS DE SEMANA

#df_definitivo['week_day']=pd.to_datetime(df_definitivo['crash_date']).dt.dayofweek

#df_semana_siniestros= df_definitivo.groupby('week_day')[['number_of_crashes']].sum().reset_index()

#df_semana_siniestros['week_day'].replace([0,1,2,3,4,5,6],['lunes','martes','miércoles','jueves','viernes','sábado','domingo'],inplace=True)

fig = px.bar(df_semana_siniestros, x="week_day", y=["number_of_crashes"], title="Siniestros por día de la semana")

st.plotly_chart(fig, use_container_width=True)

######### abierto por covid

#df_week_lockdown=df_definitivo[['lockdown','week_day','number_of_crashes']]


#df_week_lockdown.groupby(['lockdown','week_day']).mean().unstack('week_day').plot.bar(figsize=(20,5),color=(0.2, 0.2, 0.2, 0.2),edgecolor='grey')

################ DEEP-DIVE COVID
st.title("Deep-dive en un mundo post-pandémico")


image = Image.open('/content/drive/MyDrive/Colab Notebooks/images/empty_city_caratula.jpg')
st.image(image, caption='Times Square durante la pandemia.') 
 




### Gráfico semana
st.markdown("La cantidad de accidentes tanto los días de semana, como los sábados y domingos se redujeron un 50% vs. los períodos pre-pandémicos.")

image = Image.open('/content/drive/MyDrive/Colab Notebooks/images/Evolucion_semanal_covid.png')
st.image(image)


st.markdown("El orden de los principales factores que contribuyen a los accidentes no tuvieron cambios significativos .")

fig = make_subplots(rows=1, cols=2, shared_yaxes=False)
fig.add_trace(go.Bar(x=df_motivos_grouped_aux_pre["number_of_crashes"],
                     y=df_motivos_grouped_aux_pre["contributing_factor_vehicle"],orientation = 'h',name="Accidentes - Pre"),
              1, 1)
fig.add_trace(go.Bar(x=df_motivos_grouped_aux_post["number_of_crashes"], y=df_motivos_grouped_aux_post["contributing_factor_vehicle"],orientation = 'h',name="Accidentes - Post"),
              1, 2)

fig.update_layout(height=600, width=1000, title_text="Factores contributivos - Cant. Accidentes Pre y Post-Lockdown")
st.plotly_chart(fig, use_container_width=True)

st.markdown("Sin embargo, los factores causantes de accidentes que involucraron muertos, sí muestran un cambio de comportamiento.")


fig = make_subplots(rows=1, cols=2, shared_yaxes=False)

fig.add_trace(go.Bar(x=df_motivos_grouped_aux_pre["number_of_persons_killed"],
                     y=df_motivos_grouped_aux_pre["contributing_factor_vehicle"],orientation = 'h',name="Muertes - Pre"),
              1, 1)
fig.add_trace(go.Bar(x=df_motivos_grouped_aux_post["number_of_persons_killed"], y=df_motivos_grouped_aux_post["contributing_factor_vehicle"],orientation = 'h',name="Muertes - Post"),
              1, 2)

fig.update_layout(height=600, width=1000, title_text="Factores contributivos - Muertes Pre y Post-Lockdown")
st.plotly_chart(fig, use_container_width=True)

############ lockdown vs lockdown-viernes

st.markdown("Haciendo foco en los viernes, podemos ver que el comportamiento es similar al resto de los días de la semana, los mayores accidentes problamente surgen de una mayor circulación.")

fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
fig.add_trace(go.Bar(x=df_motivos_grouped_aux_post["number_of_crashes"],
                     y=df_motivos_grouped_aux_post["contributing_factor_vehicle"],orientation = 'h',name="Accidentes - Post"),
              1, 1)
fig.add_trace(go.Bar(x=df_motivos_grouped_aux_post_viernes["number_of_crashes"], y=df_motivos_grouped_aux_post_viernes["contributing_factor_vehicle"],orientation = 'h',name="Accidentes - Post - Viernes"),
              1, 2)
fig.update_layout(height=600, width=1000, title_text="Factores contributivos - Cant. Accidentes Post-Lockdown y Post-Lockdown viernes")
fig.show()

st.markdown("El cambio de comportamiento de los conductores se observa también observando las diferentes temporadas del año.")

image = Image.open('/content/drive/MyDrive/Colab Notebooks/images/Evolucion_season_covid.png')
st.image(image)
############ lockdown vs lockdown-invierno

st.markdown("El cambio de comportamiento de los conductores se observa también observando las diferentes temporadas del año.")

fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

fig.add_trace(go.Bar(x=df_motivos_grouped_aux_post["number_of_crashes"],
                     y=df_motivos_grouped_aux_post["contributing_factor_vehicle"],orientation = 'h',name="Accidentes - Post"),
              1, 1)
fig.add_trace(go.Bar(x=df_motivos_grouped_post_winter["number_of_crashes"], y=df_motivos_grouped_post_winter["contributing_factor_vehicle"],orientation = 'h',name="Accidentes - Post - Invierno"),
              1, 2)

fig.update_layout(height=600, width=1000, title_text="Factores contributivos - Cant. Accidentes Post-Lockdown y Post-Lockdown  Invierno")
fig.show()

##### ETL PARA CALCULADO KPI's
####Si el archivo es el descargado como csv, se debe hacer esto, sino comenta el codigo el cambio a datetime
data = df3


data.columns = map(str.lower, data.columns) #cambiar nombres de las columnas por minusculas
d2=data.columns.str.replace(' ', '_')#reemplazar espacios por guion bajo, es un index type
d2=d2.to_list()
data.columns=d2 #cambiar por valores de la lista
#
data['crash_date']=pd.to_datetime(data['crash_date'])
data = data.sort_values('crash_date',ascending=True)

################DASHBOARD#############################
#Esto solo es por joder, se ve lindi si le hacemos click en la seccion de menu


# dashboard title
st.title("Real-Time Data Science Dashboard")
image3 = Image.open('/content/drive/MyDrive/Colab Notebooks/images/Data-Science-1.jpg')
st.image(image3, caption='')

st.title("¿Cómo generar políticas públicas data-driven para reducir la siniestralidad en este nuevo contexto? ")
st.markdown("Con las siguientes herramientas, será posible dimensionar la cantidad de personas directamente afectadas por los siniestros en un determinado período, y preparar indicadores que nos permitan posteriormente calcular cuantas personas se verían beneficiadas por politicas precautorias en el sector de Seguridad Vial.")
#Seleccion de fecha

start_date = st.date_input("Seleccionar fecha de inicio",min_value = datetime.date(2012, 8, 26), max_value= datetime.date.today(), value=datetime.date.today())
end_date = st.date_input("Seleccionar último dia de análisis",min_value = datetime.date(2012, 8, 26), max_value= datetime.date.today(), value=datetime.date.today()) #ver si esta actualizando siempre!
filter=data.copy() #voy a hacer un campo de tipo de fecha para poder hacer la mascara, prefiero hacerlo con copia
filter['crash_date']=filter['crash_date'].dt.date
filter=filter[(filter['crash_date']>= (start_date)) & (filter['crash_date']<=(end_date))]

#Calculo de dias/anhos/deltas
deltat=(end_date-start_date)
dias=deltat.days
anhos=round(deltat.days/365,1)
future=datetime.date.today()+deltat


#Para el analisis por ahora solo se tiene en cuenta numero de accidendentes, personas heridas y muertas
r = filter.groupby('crash_date').agg({'crash_date': ['count'], 'number_of_persons_injured': ['sum'], 'number_of_persons_killed': ['sum'] }).reset_index()
r.columns = ['crash_date','number_of_crashes','number_of_persons_injured','number_of_persons_killed']


d=st.markdown(f'Tiempo total en **dias**= **_{dias}_**')
a=st.markdown(f'Tiempo total en **años**= **_{anhos}_**')
# COSTOS ECONÓMICOS
death=1750000
injured=26550
car_damage=4700    

#resumen de los datos en forma de metricas
accidentes=r.number_of_crashes.sum()
heridos=r.number_of_persons_injured.sum()
muertos=r.number_of_persons_killed.sum()
costo_economico=round((car_damage*accidentes+injured*heridos+muertos*death)/1000000,2)


col1, col2, col3, col4 = st.columns(4)
col1.metric("Accidentes", value=accidentes)
col2.metric("Heridos", value=heridos)
col3.metric("Muertos",  value=muertos)
col4.metric('Costo económico ($Mill.)', value=costo_economico)

#####"Calculadora", intento de KPI v0.0.1
##Reducir x% los choques en y anhos respecto a z cantidad de tiempo

st.header(f"KPIs objetivo sobre el periodo de tiempo seleccionado:")

porcentaje=st.number_input('% deseado de reduccion de accidentes',min_value=0.0, max_value=100.0, value=0.0)

calculo=st.button('Click para calcular')

if calculo:
    st.markdown(f'Un {porcentaje} % representa una _disminucion_ de {accidentes*porcentaje/100} accidentes.')
    st.markdown(f'Esta reducción de accidentes tendría un impacto positívo en la economía de {round(costo_economico*porcentaje/100,1)} millones de USD.')
    st.markdown(f'Esta meta tendria que alcanzarse en **{dias}** dias, {future}')
    
    
    st.metric("Accidentes menos por dia", value=round((accidentes*porcentaje/100)/dias,2), delta= f'-{porcentaje}%' )
image = Image.open('/content/drive/MyDrive/Colab Notebooks/images/Conclusiones_Iniciales_NYC.png')



####################### MAPA
# Map to show the physical locations of Crime for the selected day.



st.subheader('Mapa de Siniestralidad Vial')

#FECHASS
def between_date(data,start,end):
    mask = (data['crash_date'] >= start) & (data['crash_date'] <= end)
    return data.loc[mask]

dates = pd.to_datetime(df['crash_date'])
date_selection = st.sidebar.slider('Fecha:',
                            min_value= dates.max().date(),  
                            max_value= dates.max().date(),
                            value=(dates.min().date(),dates.max().date()))
df = between_date(df,str(date_selection[0]),str(date_selection[1]))

#FEATURES
st.subheader("Feature")
options = df.drop(columns=(['latitude','longitude','crash_date'])).columns.to_list()
map_selection = st.selectbox(label= 'Seleccionar datos de:',options=options)

if map_selection == 'total_killed':
    elevation_scale = 100
    min_value = float(0)
else:
    elevation_scale = 20
    min_value = float(3)

#GRAVEDAD
gravedad = st.sidebar.slider('Gravedad:',
                            min_value = min_value,
                            max_value = float(df[map_selection].max()),
                            value=(min_value,float(df[map_selection].max())))
df = df[(df[map_selection] >= gravedad[0]) & (df[map_selection] <= gravedad[1])]

column_layer = pdk.Layer(
    "ColumnLayer",
    data=df,
    get_position=["longitude", "latitude"],
    get_elevation=f"{map_selection}*10",
    elevation_scale=elevation_scale,
    radius=300,
    get_fill_color=[f"{map_selection} * 30", f"{map_selection} * 10", f"{map_selection} * 5"],
    pickable=True,
    auto_highlight=True,
)

tooltip = {
     "html": "Location: {longitude}</br> Latitude: {latitude} </br> Count: {total_victims}</br> Feature: {total_victims}",
     "style": {"background": "steelblue", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
}

r = pdk.Deck(
    column_layer,
    map_style=None,
    initial_view_state=pdk.ViewState(
            latitude=40.730610,
            longitude=-73.935242,
            zoom=10,
            pitch=30,
        ),
    tooltip=tooltip,
)

st.pydeck_chart(r)

image = Image.open('/content/drive/MyDrive/Colab Notebooks/images/Conclusiones_Iniciales_NYC.png')
st.image(image) 

st.subheader( '- La pandemia tuvo como resultado colateral una significativa caída en los accidentes de tránsito. La caída post-Lockdown es generada por menos personas movilizándose, tanto los días de semana como los sábados y domingos.')
st.subheader( '- Aunque se observaba mayor siniestralidad en verano, y los días viernes, esto ya no se observa en la actualidad. El manejo "DUI" no es de los principales causantes de accidentes. Sin embargo, identificamos a las distracciones (no determinadas) como uno de los principales motivos de colisiones entre vehículos.')
st.subheader( '- El impacto económico por la siniestralidad vial toma valores multimillonarios y afecta a cientos de personas cada año. La posibilidad de visualizar las esquinas con mayor cantidad de accidentes permite identificar áreas de mayor riesgo para destinar políticas públicas de mayor impacto.')
st.subheader( '- Identificamos áreas de interés, por ejemplo la esquina de Vanderbilt Ave y Bay St. donde se encuentra uno de los principales hospitales de Long Island, junto a una escuela y un edificio del Servicio Postal o los alrededores de Bedfort Ave y Lincoln Rd. en Brooklyn, junto a uno de los principales hospitales de emergencias, el Kings County Hospital')
st.subheader( '- Finalmente, al analizar otra linea de investigación relacionada a fechas de eventos deportivos,  nos lleva a proponer mayores controles en los alredores del Estadio de los "NY Yankees" (Bronx), al observarse un aumento en el promedio de accidentes durante esos días.')
