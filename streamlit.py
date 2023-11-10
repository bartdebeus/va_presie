# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:00:49 2023

@author: bartd
"""

##########################################################################################################
###Importeren van de packages:
##########################################################################################################

import pandas as pd
import numpy as np
import streamlit as st

import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

##########################################################################################################
###Streamlit titel en uitleg:
##########################################################################################################

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded")

st.title('Ontwikkeling van de huizenprijzen in Nederland.')
st.write('''Deze dashboard is gemaakt om de ontwikkeling van de huizenprijzen te laten zien. Er zijn
        hiervoor verschillende datasets gebruikt (zie referenties) en deze zijn samengevoegd voor de 
        verschillende figuren. Ook is er een voorspeller gemaakt, om de huizenprijs te voorspellen, 
        afhankelijk van verschillende parameters.
        ''')
st.divider()


tab1, tab2, tab3 = st.tabs(["Landkaarten", "Voorspeller", 'Analyse v/d huizenprijzen'])

##########################################################################################################
###Data inladen:
##########################################################################################################

@st.cache_data
def data_inladen(data):
    return pd.read_csv(f'{data}.csv', sep = ';')

inflatie_jaar= data_inladen('inflatie_jaar')
inflatie_maand = data_inladen('inflatie_maand')
bevolkingsstatistieken = data_inladen('bevolkingsstatistieken')
regio_data = data_inladen('regio_data')
land_data = data_inladen('land_data')
huurverhoging = data_inladen('huurverhoging')
logreg = data_inladen('logreg')
gemeentegrenzen = data_inladen('gemeentegrenzen')
land_data_jaar = pd.read_csv('Bestaande_koopwoningen__index_Nederland_02112023_103055.csv', sep = ';')

##########################################################################################################
###Perioden omzetten naar datetime:
##########################################################################################################

inflatie_jaar['Perioden'] = pd.to_datetime(inflatie_jaar['Perioden'], format='%Y-%m-%d')
inflatie_maand['Perioden'] = pd.to_datetime(inflatie_maand['Perioden'], format='%Y-%m-%d')
bevolkingsstatistieken['Perioden'] = pd.to_datetime(bevolkingsstatistieken['Perioden'], format='%Y-%m-%d')
regio_data['Perioden'] = pd.to_datetime(regio_data['Perioden'], format='%Y-%m-%d')
land_data['Perioden'] = pd.to_datetime(land_data['Perioden'], format='%Y-%m-%d')
huurverhoging['Perioden'] = pd.to_datetime(huurverhoging['Perioden'], format='%Y-%m-%d')
land_data_jaar['Perioden'] = pd.to_datetime(land_data_jaar['Perioden'], format='%Y')

##########################################################################################################
###Gemeentegrenzen toevoegen:
##########################################################################################################

url ='https://opendata.arcgis.com/datasets/620c2ab925f64ed5979d251ba7753b7f_0.geojson' 
gemeentegrenzen = gpd.read_file(url)

gemeentegrenzen = gemeentegrenzen[['Gemeentecode', 'geometry']]
gemeentegrenzen = gemeentegrenzen.rename(columns={'Gemeentecode': 'RegioS'})

##########################################################################################################
###Streamlit referentiebar:
##########################################################################################################

st.sidebar.title('Referenties')
st.sidebar.write('''Voor deze dashboard zijn verschillende datasets gebruikt, allen afkomstig van de 
                 statline website van het Centraal Bureau voor de Statistiek (CBS). Deze datasets zijn
                 opgeschoont, er is data samengevoegd en data bewerkt. Hieruit zijn de verschillende
                 datasets gekomen, waarmee de dashboard daadwerkelijk is gemaakt.
                 ''')

st.sidebar.divider()                 
st.sidebar.subheader('Regiodata')
st.sidebar.write('''De regiodata laat de grote statistische kerncijfers zien per regio. In deze dashboard is
                 is er gekozen om op gemeentelijke schaal te kijken.
                 ''')
st.sidebar.write("Klik om te bekijken op deze [link](https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=70072ned&_theme=237)")

st.sidebar.divider() 
st.sidebar.subheader('Landdata')
st.sidebar.write('''De landdata is de jaarlijkse prijsontwikkeling van huizen die in dat jaar te koop zijn
                 in Nederland. Het eerste jaar in de dataset is 1959.
                 ''')
st.sidebar.write("Klik om te bekijken op deze [link](https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=83906NED&_theme=387)")

st.sidebar.divider() 
st.sidebar.subheader('Huurverhoging')
st.sidebar.write('''Deze dataset laat de jaarlijkse huurverhoging zien. Deze data loopt sinds 1959 en is 
                 landelijk.
                 ''')
st.sidebar.write("Klik om te bekijken op deze [link](https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=70675ned&_theme=381)")

st.sidebar.divider() 
st.sidebar.subheader('Bevolkingsstatistieken')
st.sidebar.write('''Deze dataset laat de grote statistische kerncijfers zien, per aantal inwoners, per regio.
                 Hier is gekozen om alleen te kijken naar de gemeenten, en specifieke waarden die handig zijn
                 met betrekking tot de WOZ-waarde.
                 ''')
st.sidebar.write("Klik om te bekijken op deze [link](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/70072ned/table?ts=1698744931011)")

st.sidebar.divider() 
st.sidebar.subheader('Inflatiedata')
st.sidebar.write('''De inflatiedata laat de jaarlijkse en maandelijkse inflatie en stijging van de inflatie
                 zien. De kwartaaldata is verwijderd.
                 ''')
st.sidebar.write("Klik om te bekijken op deze [link](https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=70936ned&_theme=379)")

st.sidebar.divider() 
st.sidebar.subheader('Gemeentegrenzen')
st.sidebar.write('''De gemeentegrenzen zijn afkomstig vanuit een andere dataset, die over covid-19 gaat. Hieruit
                 zijn alleen de grenzen gehaald en de bijbehorende gemeente.
                 ''')
st.sidebar.write("Klik om te bekijken op deze [link](https://opendata.arcgis.com/datasets/620c2ab925f64ed5979d251ba7753b7f_0.geojson)")

st.sidebar.divider() 
st.sidebar.write('''Deze dashboard is gemaakt door Daan Jansen en Bart de Beus, voor de minor Data Science
                 aan de Hogeschool van Amsterdam.
                 ''')
                 
##########################################################################################################
###Voorspellingsmodel:
##########################################################################################################

linear_reg = pd.read_pickle(f'linear_reg.plk')

with tab2:
    st.subheader('Huizenprijs voorspellen')
    st.write('''Deze tabblad geeft een voorspeller neer om de huizenprijs te berekenen. Dit wordt gedaan aan de hand
             van verschillende parameters. Namelijk de bevolkingsdichtheid, de afstand tot een huisarts, bibliotheek
             en de afstand tot een treinstation. En tot slot wordt er gekeken naar in welke gemeente er voor een huis 
             gekeken moet worden.
             ''')

# Create a mapping dictionary
gemeentenaam_mapping = {
    'Aa en Hunze': 0, 'Aalsmeer': 1, 'Aalten': 2, 'Achtkarspelen': 3, 'Alblasserdam': 4, 'Albrandswaard': 5,
    'Alkmaar': 6, 'Almelo': 7, 'Almere': 8, 'Alphen aan den Rijn': 9, 'Alphen-Chaam': 10, 'Altena': 11, 'Ameland': 12,
    'Amersfoort': 13, 'Amstelveen': 14, 'Amsterdam': 15, 'Apeldoorn': 16, 'Arnhem': 17, 'Assen': 18, 'Asten': 19,
    'Baarle-Nassau': 20, 'Baarn': 21, 'Barendrecht': 22, 'Barneveld': 23, 'Beek': 24, 'Beekdaelen': 25, 'Beesel': 26,
    'Berg en Dal': 27, 'Bergeijk': 28, 'Bergen (L.)': 29, 'Bergen (NH.)': 30, 'Bergen op Zoom': 31, 'Berkelland': 32,
    'Bernheze': 33, 'Best': 34, 'Beuningen': 35, 'Beverwijk': 36, 'De Bilt': 37, 'Bladel': 38, 'Blaricum': 39,
    'Bloemendaal': 40, 'Bodegraven-Reeuwijk': 41, 'Boekel': 42, 'Borger-Odoorn': 43, 'Borne': 44, 'Borsele': 45,
    'Boxtel': 46, 'Breda': 47, 'Brielle': 48, 'Bronckhorst': 49, 'Brummen': 50, 'Brunssum': 51, 'Bunnik': 52,
    'Bunschoten': 53, 'Buren': 54, 'Capelle aan den IJssel': 55, 'Castricum': 56, 'Coevorden': 57, 'Cranendonck': 58,
    'Culemborg': 59, 'Dalfsen': 60, 'Dantumadiel': 61, 'Delft': 62, 'Deurne': 63, 'Deventer': 64, 'Diemen': 65,
    'Dijk en Waard': 66, 'Dinkelland': 67, 'Doesburg': 68, 'Doetinchem': 69, 'Dongen': 70, 'Dordrecht': 71,
    'Drechterland': 72, 'Drimmelen': 73, 'Dronten': 74, 'Druten': 75, 'Duiven': 76, 'Echt-Susteren': 77,
    'Edam-Volendam': 78, 'Ede': 79, 'Eemnes': 80, 'Eemsdelta': 81, 'Eersel': 82, 'Eijsden-Margraten': 83,
    'Eindhoven': 84, 'Elburg': 85, 'Emmen': 86, 'Enkhuizen': 87, 'Enschede': 88, 'Epe': 89, 'Ermelo': 90,
    'Etten-Leur': 91, 'De Fryske Marren': 92, 'Geertruidenberg': 93, 'Geldrop-Mierlo': 94, 'Gemert-Bakel': 95,
    'Gennep': 96, 'Gilze en Rijen': 97, 'Goeree-Overflakkee': 98, 'Goes': 99, 'Goirle': 100, 'Gooise Meren': 101,
    'Gorinchem': 102, 'Gouda': 103, "'s-Gravenhage": 104, 'Groningen': 105, 'Gulpen-Wittem': 106, 'Haaksbergen': 107,
    'Haarlem': 108, 'Haarlemmermeer': 109, 'Halderberge': 110, 'Hardenberg': 111, 'Harderwijk': 112,
    'Hardinxveld-Giessendam': 113, 'Harlingen': 114, 'Hattem': 115, 'Heemskerk': 116, 'Heemstede': 117,
    'Heerde': 118, 'Heerenveen': 119, 'Heerlen': 120, 'Heeze-Leende': 121, 'Heiloo': 122, 'Den Helder': 123,
    'Hellendoorn': 124, 'Hellevoetsluis': 125, 'Helmond': 126, 'Hendrik-Ido-Ambacht': 127, 'Hengelo': 128,
    "'s-Hertogenbosch": 129, 'Heumen': 130, 'Heusden': 131, 'Hillegom': 132, 'Hilvarenbeek': 133, 'Hilversum': 134,
    'Hoeksche Waard': 135, 'Hof van Twente': 136, 'Het Hogeland': 137, 'Hollands Kroon': 138, 'Hoogeveen': 139,
    'Hoorn': 140, 'Horst aan de Maas': 141, 'Houten': 142, 'Huizen': 143, 'Hulst': 144, 'IJsselstein': 145,
    'Kaag en Braassem': 146, 'Kampen': 147, 'Kapelle': 148, 'Katwijk': 149, 'Kerkrade': 150, 'Koggenland': 151,
    'Krimpen aan den IJssel': 152, 'Krimpenerwaard': 153, 'Laarbeek': 154, 'Land van Cuijk': 155, 'Landgraaf': 156,
    'Landsmeer': 157, 'Lansingerland': 158, 'Laren': 159, 'Leeuwarden': 160, 'Leiden': 161, 'Leiderdorp': 162,
    'Leidschendam-Voorburg': 163, 'Lelystad': 164, 'Leudal': 165, 'Leusden': 166, 'Lingewaard': 167, 'Lisse': 168,
    'Lochem': 169, 'Loon op Zand': 170, 'Lopik': 171, 'Losser': 172, 'Maasdriel': 173, 'Maasgouw': 174,
    'Maashorst': 175, 'Maassluis': 176, 'Maastricht': 177, 'Medemblik': 178, 'Meerssen': 179, 'Meierijstad': 180,
    'Meppel': 181, 'Middelburg': 182, 'Midden-Delfland': 183, 'Midden-Drenthe': 184, 'Midden-Groningen': 185,
    'Moerdijk': 186, 'Molenlanden': 187, 'Montferland': 188, 'Montfoort': 189, 'Mook en Middelaar': 190,
    'Neder-Betuwe': 191, 'Nederweert': 192, 'Nieuwegein': 193, 'Nieuwkoop': 194, 'Nijkerk': 195, 'Nijmegen': 196,
    'Nissewaard': 197, 'Noardeast-Fryslân': 198, 'Noord-Beveland': 199, 'Noordenveld': 200, 'Noordoostpolder': 201,
    'Noordwijk': 202, 'Nuenen, Gerwen en Nederwetten': 203, 'Nunspeet': 204, 'Oegstgeest': 205, 'Oirschot': 206,
    'Oisterwijk': 207, 'Oldambt': 208, 'Oldebroek': 209, 'Oldenzaal': 210, 'Olst-Wijhe': 211, 'Ommen': 212,
    'Oost Gelre': 213, 'Oosterhout': 214, 'Ooststellingwerf': 215, 'Oostzaan': 216, 'Opmeer': 217, 'Opsterland': 218,
    'Oss': 219, 'Oude IJsselstreek': 220, 'Ouder-Amstel': 221, 'Oudewater': 222, 'Overbetuwe': 223,
    'Papendrecht': 224, 'Peel en Maas': 225, 'Pekela': 226, 'Pijnacker-Nootdorp': 227, 'Purmerend': 228, 'Putten': 229,
    'Raalte': 230, 'Reimerswaal': 231, 'Renkum': 232, 'Renswoude': 233, 'Reusel-De Mierden': 234, 'Rheden': 235,
    'Rhenen': 236, 'Ridderkerk': 237, 'Rijssen-Holten': 238, 'Rijswijk': 239, 'Roerdalen': 240, 'Roermond': 241,
    'De Ronde Venen': 242, 'Roosendaal': 243, 'Rotterdam': 244, 'Rozendaal': 245, 'Rucphen': 246, 'Schagen': 247,
    'Scherpenzeel': 248, 'Schiedam': 249, 'Schiermonnikoog': 250, 'Schouwen-Duiveland': 251, 'Simpelveld': 252,
    'Sint-Michielsgestel': 253, 'Sittard-Geleen': 254, 'Sliedrecht': 255, 'Sluis': 256, 'Smallingerland': 257,
    'Soest': 258, 'Someren': 259, 'Son en Breugel': 260, 'Stadskanaal': 261, 'Staphorst': 262, 'Stede Broec': 263,
    'Steenbergen': 264, 'Steenwijkerland': 265, 'Stein': 266, 'Stichtse Vecht': 267, 'Súdwest-Fryslân': 268,
    'Terneuzen': 269, 'Terschelling': 270, 'Texel': 271, 'Teylingen': 272, 'Tholen': 273, 'Tiel': 274, 'Tilburg': 275,
    'Tubbergen': 276, 'Twenterand': 277, 'Tynaarlo': 278, 'Tytsjerksteradiel': 279, 'Uitgeest': 280, 'Uithoorn': 281,
    'Urk': 282, 'Utrecht': 283, 'Utrechtse Heuvelrug': 284, 'Vaals': 285, 'Valkenburg aan de Geul': 286,
    'Valkenswaard': 287, 'Veendam': 288, 'Veenendaal': 289, 'Veere': 290, 'Veldhoven': 291, 'Velsen': 292,
    'Venlo': 293, 'Venray': 294, 'Vijfheerenlanden': 295, 'Vlaardingen': 296, 'Vlieland': 297, 'Vlissingen': 298,
    'Voerendaal': 299, 'Voorschoten': 300, 'Voorst': 301, 'Vught': 302, 'Waadhoeke': 303, 'Waalre': 304,
    'Waalwijk': 305, 'Waddinxveen': 306, 'Wageningen': 307, 'Wassenaar': 308, 'Waterland': 309, 'Weert': 310,
    'Weesp': 311, 'West Betuwe': 312, 'West Maas en Waal': 313, 'Westerkwartier': 314, 'Westerveld': 315,
    'Westervoort': 316, 'Westerwolde': 317, 'Westland': 318, 'Weststellingwerf': 319, 'Weestvoorne': 320,
    'Wierden': 321, 'Wijchen': 322, 'Wijdemeren': 323, 'Wijk bij Duurstede': 324, 'Winterswijk': 325,
    'Woensdrecht': 326, 'Woerden': 327, 'De Wolden': 328, 'Wormerland': 329, 'Woudenberg': 330, 'Zaanstad': 331,
    'Zaltbommel': 332, 'Zandvoort': 333, 'Zeewolde': 334, 'Zeist': 335, 'Zevenaar' : 336, 'Zoetermeer' : 337,
    'Zoeterwoude': 338, 'Zuidplas' : 339, 'Zundert' : 340, 'Zutphen' : 341, 'Zwartewaterland' : 342,
    'Zwijndrecht' : 343, 'Zwolle' :344, 
    'Sint Anthonis' : 345, 'Langedijk': 346, 'Loppersum' : 347, 'Delfzijl': 348, 'Landerd': 349, 'Heerhugowaard':350,
    'Beemster': 351, 'Cuijk':352, 'Boxmeer':353, 'Uden':354, 'Mill en Sint Hubert' :355, 'Voorne aan Zee':356, 'Grave':357,
    'Haaren' : 358, 'Appingedam': 359}


# Your array of categorical values
original_array = np.array(['Aa en Hunze', 'Aalsmeer', 'Aalten', 'Achtkarspelen', 'Alblasserdam', 'Albrandswaard', 'Alkmaar', 'Almelo', 'Almere', 'Alphen aan den Rijn', 'Alphen-Chaam', 'Altena', 'Ameland', 'Amersfoort', 'Amstelveen', 'Amsterdam', 'Apeldoorn', 'Arnhem', 'Assen', 'Asten', 'Baarle-Nassau', 'Baarn', 'Barendrecht', 'Barneveld', 'Beek', 'Beekdaelen', 'Beesel', 'Berg en Dal', 'Bergeijk', 'Bergen (L.)', 'Bergen (NH.)', 'Bergen op Zoom', 'Berkelland', 'Bernheze', 'Best', 'Beuningen', 'Beverwijk', 'De Bilt', 'Bladel', 'Blaricum', 'Bloemendaal', 'Bodegraven-Reeuwijk', 'Boekel', 'Borger-Odoorn', 'Borne', 'Borsele', 'Boxtel', 'Breda', 'Brielle', 'Bronckhorst', 'Brummen', 'Brunssum', 'Bunnik', 'Bunschoten', 'Buren', 'Capelle aan den IJssel', 'Castricum', 'Coevorden', 'Cranendonck', 'Culemborg', 'Dalfsen', 'Dantumadiel', 'Delft', 'Deurne', 'Deventer', 'Diemen', 'Dijk en Waard', 'Dinkelland', 'Doesburg', 'Doetinchem', 'Dongen', 'Dordrecht', 'Drechterland', 'Drimmelen', 'Dronten', 'Druten', 'Duiven', 'Echt-Susteren', 'Edam-Volendam', 'Ede', 'Eemnes', 'Eemsdelta', 'Eersel', 'Eijsden-Margraten', 'Eindhoven', 'Elburg', 'Emmen', 'Enkhuizen', 'Enschede', 'Epe', 'Ermelo', 'Etten-Leur', 'De Fryske Marren', 'Geertruidenberg', 'Geldrop-Mierlo', 'Gemert-Bakel', 'Gennep', 'Gilze en Rijen', 'Goeree-Overflakkee', 'Goes', 'Goirle', 'Gooise Meren', 'Gorinchem', 'Gouda', "'s-Gravenhage", 'Groningen', 'Gulpen-Wittem', 'Haaksbergen', 'Haarlem', 'Haarlemmermeer', 'Halderberge', 'Hardenberg', 'Harderwijk', 'Hardinxveld-Giessendam', 'Harlingen', 'Hattem', 'Heemskerk', 'Heemstede', 'Heerde', 'Heerenveen', 'Heerlen', 'Heeze-Leende', 'Heiloo', 'Den Helder', 'Hellendoorn', 'Hellevoetsluis', 'Helmond', 'Hendrik-Ido-Ambacht', 'Hengelo', "'s-Hertogenbosch", 'Heumen', 'Heusden', 'Hillegom', 'Hilvarenbeek', 'Hilversum', 'Hoeksche Waard', 'Hof van Twente', 'Het Hogeland', 'Hollands Kroon', 'Hoogeveen', 'Hoorn', 'Horst aan de Maas', 'Houten', 'Huizen', 'Hulst', 'IJsselstein', 'Kaag en Braassem', 'Kampen', 'Kapelle', 'Katwijk', 'Kerkrade', 'Koggenland', 'Krimpen aan den IJssel', 'Krimpenerwaard', 'Laarbeek', 'Land van Cuijk', 'Landgraaf', 'Landsmeer', 'Lansingerland', 'Laren', 'Leeuwarden', 'Leiden', 'Leiderdorp', 'Leidschendam-Voorburg', 'Lelystad', 'Leudal', 'Leusden', 'Lingewaard', 'Lisse', 'Lochem', 'Loon op Zand', 'Lopik', 'Losser', 'Maasdriel', 'Maasgouw', 'Maashorst', 'Maassluis', 'Maastricht', 'Medemblik', 'Meerssen', 'Meierijstad', 'Meppel', 'Middelburg', 'Midden-Delfland', 'Midden-Drenthe', 'Midden-Groningen', 'Moerdijk', 'Molenlanden', 'Montferland', 'Montfoort', 'Mook en Middelaar', 'Neder-Betuwe', 'Nederweert', 'Nieuwegein', 'Nieuwkoop', 'Nijkerk', 'Nijmegen', 'Nissewaard', 'Noardeast-Fryslân', 'Noord-Beveland', 'Noordenveld', 'Noordoostpolder', 'Noordwijk', 'Nuenen, Gerwen en Nederwetten', 'Nunspeet', 'Oegstgeest', 'Oirschot', 'Oisterwijk', 'Oldambt', 'Oldebroek', 'Oldenzaal', 'Olst-Wijhe', 'Ommen', 'Oost Gelre', 'Oosterhout', 'Ooststellingwerf', 'Oostzaan', 'Opmeer', 'Opsterland', 'Oss', 'Oude IJsselstreek', 'Ouder-Amstel', 'Oudewater', 'Overbetuwe', 'Papendrecht', 'Peel en Maas', 'Pekela', 'Pijnacker-Nootdorp', 'Purmerend', 'Putten', 'Raalte', 'Reimerswaal', 'Renkum', 'Renswoude', 'Reusel-De Mierden', 'Rheden', 'Rhenen', 'Ridderkerk', 'Rijssen-Holten', 'Rijswijk', 'Roerdalen', 'Roermond', 'De Ronde Venen', 'Roosendaal', 'Rotterdam', 'Rozendaal', 'Rucphen', 'Schagen', 'Scherpenzeel', 'Schiedam', 'Schiermonnikoog', 'Schouwen-Duiveland', 'Simpelveld', 'Sint-Michielsgestel', 'Sittard-Geleen', 'Sliedrecht', 'Sluis', 'Smallingerland', 'Soest', 'Someren', 'Son en Breugel', 'Stadskanaal', 'Staphorst', 'Stede Broec', 'Steenbergen', 'Steenwijkerland', 'Stein', 'Stichtse Vecht', 'Súdwest-Fryslân', 'Terneuzen', 'Terschelling', 'Texel', 'Teylingen', 'Tholen', 'Tiel', 'Tilburg', 'Tubbergen', 'Twenterand', 'Tynaarlo', 'Tytsjerksteradiel', 'Uitgeest', 'Uithoorn', 'Urk', 'Utrecht', 'Utrechtse Heuvelrug', 'Vaals', 'Valkenburg aan de Geul', 'Valkenswaard', 'Veendam', 'Veenendaal', 'Veere', 'Veldhoven', 'Velsen', 'Venlo', 'Venray', 'Vijfheerenlanden', 'Vlaardingen', 'Vlieland', 'Vlissingen', 'Voerendaal', 'Voorschoten', 'Voorst', 'Vught', 'Waadhoeke', 'Waalre', 'Waalwijk', 'Waddinxveen', 'Wageningen', 'Wassenaar', 'Waterland', 'Weert', 'Weesp', 'West Betuwe', 'West Maas en Waal', 'Westerkwartier', 'Westerveld', 'Westervoort', 'Westerwolde', 'Westland', 'Weststellingwerf', 'Weestvoorne', 'Wierden', 'Wijchen', 'Wijdemeren', 'Wijk bij Duurstede', 'Winterswijk', 'Woensdrecht', 'Woerden', 'De Wolden', 'Wormerland', 'Woudenberg', 'Zaanstad', 'Zaltbommel', 'Zandvoort', 'Zeewolde', 'Zeist', 'Zevenaar', 'Zoetermeer', 'Zoeterwoude', 'Zuidplas', 'Zundert', 'Zutphen', 'Zwartewaterland', 'Zwijndrecht', 'Zwolle'])

with tab2:
# Dropdown to select a brand
    MunicipalityID = st.selectbox("Kies een gemeente:", original_array)
# Convert the selected brand to its numeric value
numeric_value = gemeentenaam_mapping.get(MunicipalityID)

with tab2:
    Bevolkingsdichtheid_57 = int(st.number_input('Kies de bevolkingsdichtheid: ', min_value = 22, max_value = 6712))
    AfstandTotHuisartsenpraktijk_209 = int(st.number_input('Kies de afstand tot een huisartsenpraktijk: ', min_value = 0.4, max_value = 2.8))
    AfstandTotBibliotheek_226 = int(st.number_input('Kies de afstand tot een bibliotheek: ', min_value = 0.6, max_value = 14.3))
    AfstandTotTreinstation_233 = int(st.number_input('Kies de afstand tot een treinstation: ', min_value = 1.1, max_value = 48.5))

@st.cache_resource
def predict_price(Bevolkingsdichtheid_57, AfstandTotHuisartsenpraktijk_209, AfstandTotBibliotheek_226, AfstandTotTreinstation_233, MunicipalityID):
    # Create a numpy array with the input values
    input_data = np.array([[Bevolkingsdichtheid_57, AfstandTotHuisartsenpraktijk_209, AfstandTotBibliotheek_226, AfstandTotTreinstation_233, numeric_value]])

    # Use the trained linear regression model to make predictions
    predicted_price = linear_reg.predict(input_data)

    return predicted_price[0]

predicted_price = predict_price(Bevolkingsdichtheid_57, AfstandTotHuisartsenpraktijk_209, AfstandTotBibliotheek_226, AfstandTotTreinstation_233, numeric_value)
with tab2:
    st.write(f'De voorspelde prijs is €{round(predicted_price,2)} duizend.')

##########################################################################################################
###Plotten landkaarten:
##########################################################################################################
regio_data['GemiddeldeWOZWaardeVanWoningen_98'] = regio_data['GemiddeldeWOZWaardeVanWoningen_98'].str.strip()

# Replace '.' with NaN
regio_data['GemiddeldeWOZWaardeVanWoningen_98'] = regio_data['GemiddeldeWOZWaardeVanWoningen_98'].replace('.', np.nan)

# Convert the column to float and then to integers
regio_data['GemiddeldeWOZWaardeVanWoningen_98'] = regio_data['GemiddeldeWOZWaardeVanWoningen_98'].astype(float).astype(pd.Int64Dtype())


bevolkingsstatistieken['AfstandTotBibliotheek_226'] = pd.to_numeric(bevolkingsstatistieken['AfstandTotBibliotheek_226'], errors='coerce')


with tab1:
    col1, col2 = st.columns((1.3, 0.5))
    with col1:
        st.subheader('Landkaarten.')
        st.write('''In dit tabblad staan alle landkaarten, met verschillende statistieken. Deze
             landkaarten zijn verdeeld per regionen, en er is gekozen voor gemeenten. De data staat
             er per specifiek jaartal, die kunt u aanpassen.
             ''')
    with col2:       
        st.write('')
        st.write('')
        gekozen_jaartal = st.slider('Kies een jaartal voor de plotjes: ', 2007, 2021, 2021)
    st.divider()

regio_data = regio_data.merge(gemeentegrenzen, on='RegioS', how='left')
bevolkingsstatistieken = bevolkingsstatistieken.merge(gemeentegrenzen, on = 'RegioS', how = 'left')

# Assuming you already have a GeoDataFrame (gdf) and it's correctly loaded
@st.cache_data
def plotten_map_woz(jaar, regio_data):
    merged_data_jaar = regio_data[regio_data['Year'] == jaar]
    
    gdf = gpd.GeoDataFrame(merged_data_jaar)
    gdf = gdf.dropna(subset='GemiddeldeWOZWaardeVanWoningen_98')
    gdf['GemiddeldeWOZWaardeVanWoningen_98'] = gdf['GemiddeldeWOZWaardeVanWoningen_98'].astype(int)
    
    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color='GemiddeldeWOZWaardeVanWoningen_98',
        title='WOZ-waarde van huizen',
        labels={'Gemiddelde WOZ Waarde': 'Gemiddelde WOZ Waarde'},
        hover_name='Gemeentenaam',
        projection="mercator",  # You can change the projection as needed
        color_continuous_scale = 'viridis')

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(coloraxis_colorbar=dict(title=""))
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plotten_map_afstand_bieb(jaar, bevolkingsstatistieken):
    merged_data_jaar = bevolkingsstatistieken[bevolkingsstatistieken['Year'] == jaar]
    
    gdf = gpd.GeoDataFrame(merged_data_jaar)
    gdf = gdf.dropna(subset='AfstandTotBibliotheek_226')
    gdf['AfstandTotBibliotheek_226'] = gdf['AfstandTotBibliotheek_226'].astype(float)
    
    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color='AfstandTotBibliotheek_226',
        title='Afstand tot de bibliotheek.',
        labels={'Gemiddelde WOZ Waarde': 'Gemiddelde WOZ Waarde'},
        hover_name='Gemeentenaam',
        projection="mercator",  # You can change the projection as needed
        color_continuous_scale = 'YlGnBu')

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(coloraxis_colorbar=dict(title=""))
    st.plotly_chart(fig, use_container_width=True)
    
@st.cache_data
def plotten_map_afstand_huisarts(jaar, bevolkingsstatistieken):
    merged_data_jaar = bevolkingsstatistieken[bevolkingsstatistieken['Year'] == jaar]
    
    gdf = gpd.GeoDataFrame(merged_data_jaar)
    gdf = gdf.dropna(subset='AfstandTotHuisartsenpraktijk_209')
    gdf['AfstandTotHuisartsenpraktijk_209'] = gdf['AfstandTotHuisartsenpraktijk_209'].astype(float)
    
    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color='AfstandTotHuisartsenpraktijk_209',
        title='Afstand tot de huisarts.',
        labels={'Gemiddelde WOZ Waarde': 'Gemiddelde WOZ Waarde'},
        hover_name='Gemeentenaam',
        projection="mercator",  # You can change the projection as needed
        color_continuous_scale = 'YlOrRd')

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(coloraxis_colorbar=dict(title=""))
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plotten_map_bevolkingsdichtheid(jaar, bevolkingsstatistieken):
    merged_data_jaar = bevolkingsstatistieken[bevolkingsstatistieken['Year'] == jaar]
    
    gdf = gpd.GeoDataFrame(merged_data_jaar)
    gdf = gdf.dropna(subset='Bevolkingsdichtheid_57')
    gdf['Bevolkingsdichtheid_57'] = gdf['Bevolkingsdichtheid_57'].astype(float)
    
    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color='Bevolkingsdichtheid_57',
        title='Bevolkingsdichtheid.',
        labels={'Gemiddelde WOZ Waarde': 'Gemiddelde WOZ Waarde'},
        hover_name='Gemeentenaam',
        projection="mercator",  # You can change the projection as needed
        color_continuous_scale = 'cividis')

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(coloraxis_colorbar=dict(title=""))
    st.plotly_chart(fig, use_container_width=True)


with tab1:
    col_map_1, col_gap_1, col_map_2 = st.columns((0.5,0.05, 0.5))
    with col_map_1:
        plotten_map_woz(gekozen_jaartal, regio_data)
    with col_map_2:
        plotten_map_afstand_bieb(gekozen_jaartal, bevolkingsstatistieken)
        
with tab1:
    st.divider()
    col_map_3, col_gap_2, col_map_4 = st.columns((0.5, 0.05, 0.5)) 
    with col_map_3:
        plotten_map_afstand_huisarts(gekozen_jaartal, bevolkingsstatistieken)
    with col_map_4:
        plotten_map_bevolkingsdichtheid(gekozen_jaartal, bevolkingsstatistieken)

##########################################################################################################
###Plotten van Daan:
##########################################################################################################

with tab3:
    col3, col4 = st.columns((1.3, 0.5))
    with col3:
        st.subheader('Visualisatie v/d huizenprijzen.')
        st.write('''Over de jaren heen zijn er grote veranderingen geweest in de WOZ-waarde van koopwoningen.
                 In deze tabblad wordt er gekeken naar deze prijsveranderingen, en wordt die ook vergeleken met
                 inflatie en de prijsveranderingen van huurwoningen.
                 ''')
    with col4:       
        st.write('')
        st.write('')
        gekozen_jaartal_daan = st.slider('Kies een jaartal voor de plotjes: ', 2007, 2021, 2020)
    st.divider()
    

#Omzetten naar numerieke waarden.
regio_data['GemiddeldeWOZWaardeVanWoningen_98'] = pd.to_numeric(regio_data['GemiddeldeWOZWaardeVanWoningen_98'], errors='coerce')

# Group data by "Provincienaam" and calculate the mean


# Function to plot WOZ-waarde per provincie
@st.cache_resource
def plot_woz_waarde(gekozen_jaartal_daan):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    data_plot = regio_data[regio_data['Year'] == gekozen_jaartal_daan]
    grouped_data = data_plot.groupby("Provincienaam")["GemiddeldeWOZWaardeVanWoningen_98"].mean().reset_index()
    grouped_data = grouped_data.sort_values(by="GemiddeldeWOZWaardeVanWoningen_98")    
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Provincienaam", y="GemiddeldeWOZWaardeVanWoningen_98", data=grouped_data, palette="viridis")
    plt.xlabel('Provincie')
    plt.ylabel('Gemiddelde WOZ-waarde van woningen')
    plt.title(f'Gemiddelde WOZ-waarde per provincie in {gekozen_jaartal_daan}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot()

# Function to plot bevolkingsdichtheid per provincie in 2022
@st.cache_resource
def plot_bevolkingsdichtheid_2022(gekozen_jaartal_daan):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    bevolkingsstatistieken_jaar = bevolkingsstatistieken[bevolkingsstatistieken['Year'] == gekozen_jaartal_daan]
    # Group data by "Provincienaam" and calculate the mean
    grouped_data2 = bevolkingsstatistieken_jaar.groupby("Provincienaam")["Bevolkingsdichtheid_57"].mean().reset_index()

    # Sort the data by average population density
    grouped_data2 = grouped_data2.sort_values(by="Bevolkingsdichtheid_57")
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Provincienaam", y="Bevolkingsdichtheid_57", data=grouped_data2, palette="viridis")
    plt.xlabel('Provincie')
    plt.ylabel('Gemiddelde bevolkingsdichtheid')
    plt.title(f'Gemiddelde bevolkingsdichtheid per provincie in {gekozen_jaartal_daan}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot()

# Function to plot population per provincie in 2022
@st.cache_resource
def plot_bevolking_2022(gekozen_jaartal_daan):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    bevolkingsstatistieken_jaar = bevolkingsstatistieken[bevolkingsstatistieken['Year'] == gekozen_jaartal_daan]
    # Assuming 'TotaleBevolking_1' is a numerical column
    bevolkingsstatistieken_jaar['TotaleBevolking_1'] = pd.to_numeric(bevolkingsstatistieken_jaar['TotaleBevolking_1'], errors='coerce')
    # Group data by "Provincienaam" and calculate the sum
    grouped_data3 = bevolkingsstatistieken_jaar.groupby("Provincienaam")["TotaleBevolking_1"].sum().reset_index()
    # Sort the data by total population
    grouped_data3 = grouped_data3.sort_values(by="TotaleBevolking_1")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Provincienaam", y="TotaleBevolking_1", data=grouped_data3, palette="viridis")
    plt.xlabel('Provincie')
    plt.ylabel('Bevolking (in miljoen)')
    plt.title(f'Bevolking per provincie in {gekozen_jaartal_daan}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot()

# Filter DataFrame based on the year 2022
bevolkingsstatistieken_2022 = bevolkingsstatistieken[bevolkingsstatistieken['Year'] == gekozen_jaartal_daan]


with tab3:
    col_map_1, col_gap_1, col_map_2 = st.columns((0.5, 0.05, 0.5)) 
    with col_map_1:
        # Create a Streamlit app
        st.title('Informatie per provincie')
        # Dropdown menu for selecting the graph
        selected_graph = st.selectbox('Kies een grafiek:', ['Gemiddelde WOZ-waarde per provincie', 'Gemiddelde bevolkingsdichtheid per provincie', 'Bevolking per provincie'])
        # Display the selected graph
        if selected_graph == 'Gemiddelde WOZ-waarde per provincie':       
            plot_woz_waarde(gekozen_jaartal_daan)
        elif selected_graph == 'Gemiddelde bevolkingsdichtheid per provincie':
            plot_bevolkingsdichtheid_2022(gekozen_jaartal_daan)
        elif selected_graph == 'Bevolking per provincie':
            plot_bevolking_2022(gekozen_jaartal_daan)
    with col_map_2:
        st.title("Bevolkingsdichtheid vs. Totale Bevolking")
        selected_province = st.selectbox("Kies een provincie", ['Nederland'] + list(bevolkingsstatistieken_2022['Provincienaam'].unique()))


# Update scatter plot based on the selected province
if selected_province == 'Nederland':
    scatter_fig = px.scatter(bevolkingsstatistieken_2022, x="Bevolkingsdichtheid_57", y="TotaleBevolking_1",
                             opacity=0.4,
                             labels={'Bevolkingsdichtheid_57': 'Bevolkingsdichtheid',
                                     'TotaleBevolking_1': 'Totale Bevolking'},
                             hover_data={"Gemeentenaam": True},
                             title='Bevolkingsdichtheid vs. Totale Bevolking')
else:
    province_data = bevolkingsstatistieken_2022[bevolkingsstatistieken_2022['Provincienaam'] == selected_province]
    scatter_fig = px.scatter(province_data, x="Bevolkingsdichtheid_57", y="TotaleBevolking_1",
                             opacity=0.4,
                             labels={'Bevolkingsdichtheid_57': 'Bevolkingsdichtheid',
                                     'TotaleBevolking_1': 'Totale Bevolking'},
                             hover_data={"Gemeentenaam": True},
                             title='Bevolkingsdichtheid vs. Totale Bevolking')

# Update the scatter plot
scatter_fig.update_layout(height=600, showlegend=False)

# Make all points red
scatter_fig.update_traces(marker=dict(color='red'))
# Add grid to the scatter plot
scatter_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
scatter_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
# Display the scatter plot

with tab3:
    with col_map_2:
        st.plotly_chart(scatter_fig)
    st.divider()
        
        
################################################################################################################################################


with tab3:
    col_map_3, col_gap_2, col_map_4 = st.columns((0.5, 0.05, 0.5)) 
  
@st.cache_data
def verkoop_plot():
    sns.lineplot(data=land_data, y="GemiddeldeVerkoopprijs_7", x="Perioden")
    plt.xlabel('Jaar')
    plt.ylabel('Gemiddelde verkoopprijs')
    plt.title('Gemiddelde verkoopprijs door de jaren')
    plt.grid()
    st.pyplot()

@st.cache_data
def cumsum_plot(inflatie_jaar, huurverhoging):   
    # Assuming 'Perioden' is the column containing the years
    # Convert 1963 to a datetime object
    filter_date = pd.to_datetime('1963-01-01')
    # Filter data for the period 1963 and onwards
    inflatie_jaar = inflatie_jaar[inflatie_jaar['Perioden'] >= filter_date]
    huurverhoging = huurverhoging[huurverhoging['Perioden'] >= filter_date]
    # Cumulative sum for inflation_year
    inflatie_jaar['Cumulatief'] = inflatie_jaar['JaarmutatieCPI_1'].cumsum()
    inflatie_jaar['Dataset'] = 'Inflatie'
    # Cumulative sum for rent increase
    huurverhoging['Cumulatief'] = huurverhoging['Huurverhoging_1'].cumsum()
    huurverhoging['Dataset'] = 'Huurverhoging'
    # Combine the datasets
    combined_data = pd.concat([inflatie_jaar, huurverhoging])
    # Line plot for the combined data
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_data, y="Cumulatief", x="Perioden", hue="Dataset")
    plt.xlabel('Jaren')
    plt.ylabel('Procenten gegroeid cumulatief')
    plt.title('Cumulatieve som van de stijging van de inflatie en huurverhoging over de tijd (vanaf 1964)')
    plt.legend(title='Dataset')
    plt.grid()
    st.pyplot()

@st.cache_data
def cumsum_plot2():
    start_date = pd.to_datetime('1996-01-01')
    land_data_jaar_filtered = land_data_jaar[land_data_jaar['Perioden'] >= start_date].copy()
    land_data_jaar_filtered['Cumulatief'] = land_data_jaar_filtered['Prijsindex verkoopprijzen/Ontwikkeling'].cumsum()
    inflatie_jaar_filtered = inflatie_jaar[inflatie_jaar['Perioden'] >= start_date].copy()
    inflatie_jaar_filtered['Cumulatief'] = inflatie_jaar_filtered['JaarmutatieCPI_1'].cumsum()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=land_data_jaar_filtered, y="Cumulatief", x="Perioden", label='Verkoopprijzen')
    sns.lineplot(data=inflatie_jaar_filtered, y="Cumulatief", x="Perioden", label='Inflatie')
    plt.xlabel('Jaren')
    plt.ylabel('Procenten gegroeid cumulatief')
    plt.title('Cumulatieve som van inflatie en de verhoging van de woonprijzen sinds 1996 (in %)')
    plt.legend(title='Dataset')
    plt.grid()
    st.pyplot()


##################################################################################################################################
###Boxplot basisdata:
##################################################################################################################################  

# Assuming you already have a GeoDataFrame (gdf) and it's correctly loaded
@st.cache_data
def boxplot1_wozwaarde(gekozen_jaartal_daan, regio_data):
    merged_data_jaar = regio_data[regio_data['Year'] == gekozen_jaartal_daan]
    
    merged_data_jaar = merged_data_jaar.dropna(subset='GemiddeldeWOZWaardeVanWoningen_98')
    merged_data_jaar['GemiddeldeWOZWaardeVanWoningen_98'] = merged_data_jaar['GemiddeldeWOZWaardeVanWoningen_98'].astype(int)
    
    fig = px.box(merged_data_jaar, x='Provincienaam', y='GemiddeldeWOZWaardeVanWoningen_98', title='Boxplot v/d WOZ-waarde per provincie',
                 hover_data=['Gemeentenaam'])
    fig.update_yaxes(title_text='WOZ-waarde')
    st.plotly_chart(fig)

@st.cache_data
def plotten_map_afstand_bieb(gekozen_jaartal_daan, bevolkingsstatistieken):
    merged_data_jaar = bevolkingsstatistieken[bevolkingsstatistieken['Year'] == gekozen_jaartal_daan]
    
    merged_data_jaar = merged_data_jaar.dropna(subset='AfstandTotBibliotheek_226')
    merged_data_jaar['AfstandTotBibliotheek_226'] = merged_data_jaar['AfstandTotBibliotheek_226'].astype(float)
    
    fig = px.box(merged_data_jaar, x='Provincienaam', y='AfstandTotBibliotheek_226', title='Boxplot v/d afstand tot de bibliotheek per provincie',
                 hover_data=['Gemeentenaam'])
    fig.update_yaxes(title_text='Afstand tot de bibliotheek')
    st.plotly_chart(fig)
    
@st.cache_data
def plotten_map_afstand_huisarts(gekozen_jaartal_daan, bevolkingsstatistieken):
    merged_data_jaar = bevolkingsstatistieken[bevolkingsstatistieken['Year'] == gekozen_jaartal_daan]
    
    merged_data_jaar = merged_data_jaar.dropna(subset='AfstandTotHuisartsenpraktijk_209')
    merged_data_jaar['AfstandTotHuisartsenpraktijk_209'] = merged_data_jaar['AfstandTotHuisartsenpraktijk_209'].astype(float)
    
    fig = px.box(merged_data_jaar, x='Provincienaam', y='AfstandTotHuisartsenpraktijk_209', title='Boxplot v/d afstand tot de huisarts per provincie',
                 hover_data=['Gemeentenaam'])
    fig.update_yaxes(title_text='Afstand tot de huisarts')
    st.plotly_chart(fig)

@st.cache_data
def plotten_map_bevolkingsdichtheid(gekozen_jaartal_daan, bevolkingsstatistieken):
    merged_data_jaar = bevolkingsstatistieken[bevolkingsstatistieken['Year'] == gekozen_jaartal_daan]
    
    merged_data_jaar = merged_data_jaar.dropna(subset='Bevolkingsdichtheid_57')
    merged_data_jaar['Bevolkingsdichtheid_57'] = merged_data_jaar['Bevolkingsdichtheid_57'].astype(float)
    
    fig = px.box(merged_data_jaar, x='Provincienaam', y='Bevolkingsdichtheid_57', title='Boxplot v/d bevolkingsdichtheid per provincie',
                 hover_data = ['Gemeentenaam'])
    fig.update_yaxes(title_text='Bevolkingsdichtheid')
    st.plotly_chart(fig)


##################################################################################################################################
###Boxplot basisdata:
################################################################################################################################## 


with tab3:
    with col_map_3:
        # Create a Streamlit app
        st.title("Huizenmarkt door de jaren")
        # Dropdown to select the plot
        selected_plot = st.selectbox("Kies een plot", ["Gemiddelde Verkoopprijs", "Cumulatieve som van de stijging van de inflatie en huurverhoging over de tijd (vanaf 1964)", 
                                                     "Cumulatieve som van inflatie en de verhoging van de woonprijzen sinds 1996 (in %)"])
        if selected_plot == "Gemiddelde Verkoopprijs":
            verkoop_plot()
        elif selected_plot ==  "Cumulatieve som van de stijging van de inflatie en huurverhoging over de tijd (vanaf 1964)":
            cumsum_plot(inflatie_jaar, huurverhoging)
        elif selected_plot == "Cumulatieve som van inflatie en de verhoging van de woonprijzen sinds 1996 (in %)":
            cumsum_plot2()
            
    with col_map_4:
        st.title(f"Boxplots voor de waarden in de kaarten in {gekozen_jaartal_daan}")
        
        selected_plot1 = st.selectbox("Kies een boxplot", ["WOZ-waarde", "Afstand bibliotheek", 'Afstand huisarts', 'Bevolkingsdichtheid'])
        if selected_plot1 == 'WOZ-waarde':
            boxplot1_wozwaarde(gekozen_jaartal_daan, regio_data)
        elif selected_plot1 == "Afstand bibliotheek":
            plotten_map_afstand_bieb(gekozen_jaartal_daan, bevolkingsstatistieken)
        elif selected_plot1 == 'Afstand huisarts':
            plotten_map_afstand_huisarts(gekozen_jaartal_daan, bevolkingsstatistieken)
        elif selected_plot1 == 'Bevolkingsdichtheid':
            plotten_map_afstand_huisarts(gekozen_jaartal_daan, bevolkingsstatistieken)























































































































