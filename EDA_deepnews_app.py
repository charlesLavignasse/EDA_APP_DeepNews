import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import EngFormatter
sns.reset_orig()

#on importe nos deux datasets
if st.checkbox('Voir le QR_Code :'):
    st.image('qr_code.png')

campaigns = pd.read_csv("campaigns.csv",error_bad_lines=False)

digest_topics = pd.read_csv("Digest Topic.csv",error_bad_lines=False)
digest_topics = digest_topics.replace("Economy", 'Business')
#là on crée le titre de la page
st.title("Exploration des métriques")

#On selectionne dans notre dataset les données qui nous interessent
digest = campaigns[campaigns['List']== 'Deepnews Digest']
digest = digest[digest['Send Weekday']=='Friday']

#formattage de la colonne 'Send Date' en datetime
digest['Send Date'] = pd.to_datetime(digest["Send Date"],format = "%b %d, %Y %H:%M %p")
#on enlève les données inutiles
digest.drop([186,191,200,210], inplace = True)

#copie du dataset
digest_final = digest.copy()
mylambda= lambda x: x.strip('%')
digest_final['Click Rate']=digest_final['Click Rate'].apply(mylambda)
digest_final['Open Rate']=digest_final['Open Rate'].apply(mylambda)
digest_final['Open Rate']=digest_final['Open Rate'].astype('float64')
digest_final['Click Rate']=digest_final['Click Rate'].astype('float64')


dateDigest = digest_final["Send Date"]

st.header('voir le dataset des métriques')
nombre_lignes_a_visualiser = st.slider("Nombre de lignes à  visualiser",0,25,5)
st.write(digest_final.head(nombre_lignes_a_visualiser))




#création d'une fonction de plotting

def LinePlotTime(parameter, Parameter_name, dataset, title_name):
    fig, axes = plt.subplots(figsize = (15,8))
    sns.lineplot(x = dateDigest, y = parameter, data = dataset, linewidth=4, c='orangered')
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    x_label_list = ['Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre','Décembre']
    axes.set_xticklabels(x_label_list)
    plt.xlabel("Date d'envoi",fontsize=20)
    plt.ylabel(Parameter_name,fontsize=20)
    plt.xlim(xmin=("2019-06-15"))
    plt.xticks(rotation=30)
    plt.title(title_name, fontsize=25)
    plt.show()


def LinePlotTimePercent(parameter, Parameter_name, dataset, title_name,moy_indus,ymin,ymax):
    fig, axes = plt.subplots(figsize = (15,8))
    ax = sns.lineplot(x= dateDigest, y = parameter, data = dataset, linewidth=4, c='orangered', label = "DeepNews")
    ax1=sns.lineplot(x='Send Date', y=moy_indus, data=digest, linewidth=2.5, c='navy', label="Moyenne du secteur")
    ax.lines[1].set_linestyle("--")
    plt.ylim(ymin,ymax)
    formatter0 = EngFormatter(unit='%')
    ax.yaxis.set_major_formatter(formatter0)
    x_label_list = ['Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre','Décembre']
    axes.set_xticklabels(x_label_list)
    plt.xlim(xmin=("2019-06-15"))
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    plt.xlabel("Date d'envoi",fontsize=20)
    plt.ylabel(Parameter_name,fontsize=20)
    plt.xticks(rotation=30)
    plt.title(title_name, fontsize=25)


digest["Reactity Rate"]= digest["Unique Clicks"]*100/digest["Unique Opens"]

def reactity_plot():
    fig, ax = plt.subplots(figsize = (15,8))
    ax=sns.lineplot(x = 'Send Date', y = 'Reactity Rate', data = digest, linewidth=4, c='#FF0700', label = "DeepNews")
    ax1=sns.lineplot(x='Send Date', y=20.767494356659142, data=digest, linewidth=2.5, c='navy', label="Moyenne du secteur")
    ax.lines[1].set_linestyle("--")
    ax.legend(fontsize=18)
    plt.ylim(0,40)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    formatter0 = EngFormatter(unit='%')
    ax.yaxis.set_major_formatter(formatter0)
    plt.xlabel("Date d'envoi",fontsize=20)
    plt.ylabel('Taux de réactité',fontsize=20)
    plt.xticks(rotation=0)
    plt.title("Evolution du taux de réactivité", fontsize=27)
    x_label_list = ['Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre','Décembre']
    ax.set_xticklabels(x_label_list)

    plt.xlim(xmin=("2019-06-15"))

    plt.tight_layout()
    #pour le chargement des graphiques sur l'ordinateur
    # plt.savefig("click_rate.png")
    # files.download("click_rate.png")
    plt.show()

#On crée la selectbox pour les métriques
metrique_temps = st.selectbox('Quelle métrique veux-tu représenter',("Destinataires","Taux d'ouverture", 'Taux de clic','Taux de réactivité'))

if metrique_temps == "Destinataires":
    #on représente le nombre de receveurs en fonction du temps
    plot_totalR = LinePlotTime('Total Recipients','Nombre de destinataires', digest_final,"Evolution du nombre de destinataires en fonction du temps")
    st.write("Evolution du nombre de destinataires recevant la newsletter")
    st.pyplot(plot_totalR)

elif metrique_temps == "Taux d'ouverture":
    plot_OpenR = LinePlotTimePercent('Open Rate', "Taux d'ouverture", digest_final,"Evolution du taux d'ouverture en fonction du temps",22.15, 0,80 )
    st.write("Evolution du taux d'ouverture")
    st.pyplot(plot_OpenR)

elif metrique_temps == 'Taux de clic':
    plot_clicR = LinePlotTimePercent('Click Rate','taux de clic', digest_final,"Evolution du taux de clic en fonction du temps",4.6, 0,17)
    st.write("Evolution du taux de clic en fonction du temps")
    st.pyplot(plot_clicR)
else :
    reactity_plot = reactity_plot()
    st.write("Evolution du Taux de réactivité")
    st.pyplot(reactity_plot)


#on intègre les thèmes dans notre jeu de données
dateDigest = digest_final["Send Date"]
digest_final = digest_final.reset_index(drop = True)
digest_final = digest_final.drop([0,1,2,3])
digest_final = digest_final.reset_index(drop= True)

digest2=pd.concat([digest_final,digest_topics],axis = 1)
digest_theme=digest2
theme = digest_theme["Thème"]




def barplots(parameter, Parameter_name, title_name):
    fig, ax = plt.subplots(figsize = (20, 7))
    sns.barplot(x = theme, y = parameter, data = digest_theme )
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.xlabel("Thème",fontsize=20)
    x_label_list = ['Politique', 'Tech', 'Réseau social', 'Santé', 'Affaires', 'Relations' + '\n'+ "Internationales"]
    ax.set_xticklabels(x_label_list)
    plt.ylabel(Parameter_name,fontsize=20)
    plt.xticks(rotation=0)
    plt.title(title_name, fontsize=25)

st.title("Métriques en fonction des thèmes")
if st.checkbox("voir les différents thèmes"):
    st.write(pd.DataFrame(digest_theme["Thème"].value_counts()))

#st.write(digest_theme)
st.header("Représentation des métriques en fonction du thème de la newsletter")
barTheme = st.selectbox("Quelle métrique veux-tu représenter ?", ("Taux d'ouverture", "taux de clic", "Clics uniques"))
if barTheme == "Taux d'ouverture":
    barplot_openR = barplots('Open Rate', "Taux d'ouverture", "Taux d'ouverture en fonction du thème")
    st.write("Taux d'ouverture en fonction du thème")
    st.pyplot(barplot_openR)
elif barTheme == "taux de clic":
    barplot_ClickR = barplots('Click Rate', "taux de clic", 'taux de clic en fonction du thème')
    st.write("taux de clic en fonction du thème")
    st.pyplot(barplot_ClickR)
else :
    barplot_UClick = barplots("Unique Clicks", "Clics uniques", 'taux de clic uniques en fonction du thème')
    st.write("Taux de clics en fonction du thème")
    st.pyplot(barplot_UClick)



st.header("Représentation des désinscriptions en fonction du thème et de la newsletter")

def scatterthing(x, y, hue,xlabel,ylabel, title):
    fig, ax = plt.subplots(figsize=(20,10))
    sns.scatterplot(digest_theme["Unsubscribes"].sort_values(),digest_theme['Title'], hue = digest_theme['Thème'], s = 300 )

    # plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    # formatter0 = EngFormatter(unit='%')
    # ax.yaxis.set_major_formatter(formatter0)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.xticks(fontsize = 17)
    plt.title(title, fontsize=27)
    plt.legend(fontsize = 15)
    plt.tight_layout()

scat_Uns = scatterthing(digest_theme["Unsubscribes"],digest_theme['Title'],digest_theme['Thème'],"Unsubscribers",'Digest issue',"Unsubscribers by digest by thème" )

st.header("voir le scatterplot")
st.pyplot(scat_Uns)



st.header("Visualisations des subscribers")
digest_theme['New Subscribers'] = 0
for i in digest_theme.index:
  if i == 0:
    digest_theme.loc[i,'New Subscribers'] = 0
  else :
    digest_theme.loc[i, 'New Subscribers'] = digest_theme.loc[i, 'Total Recipients'] - digest_theme.loc[i - 1, 'Total Recipients']

#Date = digest_theme_complete['Send Date']
#New_Subscribers = digest_theme_complete['New Subscribers']
#Unsuscribers = digest_theme_complete['Unsubscribes']
#Theme = digest_theme_complete['Thème']

def doubleLinePlot():
    fig, axes = plt.subplots(2,1, figsize = (20, 10),)

    New_sub = sns.lineplot(x = digest_theme['Send Date'], y = digest_theme['New Subscribers'],  ax = axes[0])
    Unsub = sns.lineplot(x = digest_theme   ['Send Date'], y = digest_theme['Unsubscribes'] , ax = axes[1])

    New_sub.set_xlabel("Date",fontsize=18)
    Unsub.set_xlabel("Date",fontsize=18)

    New_sub.set_ylabel("Nouveaux abonnés",fontsize=20)
    Unsub.set_ylabel("Désabonnés",fontsize=20)

    New_sub.tick_params(labelsize=15)
    Unsub.tick_params(labelsize=15)

    plt.tight_layout()



doubleplot = doubleLinePlot()
st.header("voir le double plot")
st.pyplot(doubleplot)


st.title("Analyse de la répartition de clics")

st.header('Distribution du nombre de clics utilisateurs')
st.write("Voir la distribution : ")
st.image('clic_user.png')
st.image('clic_user_zoom.png')


st.header('Représentation des éditeurs en fonction de leur catégorie')
category = pd.read_csv("publishers_list.csv", sep = ";")
publisher = pd.read_csv("publisher.csv")
publisher = publisher.drop(columns=['Unnamed: 0', 'url'])
publisher_group = publisher.groupby(by = "publisher").sum().sort_values(by = 'total_clicks', ascending = False).reset_index()
publisher_category = pd.merge(publisher_group, category, on = "publisher")
st.write('Graphe des éditeurs en fonction des clics utilisateurs : ')
nombre_publisher_slider = st.slider("Choisir le nombre d'éditeurs à visualiser :", 0,100,20)
def publi_cat():
    fig, ax = plt.subplots(figsize= (15, 5))
    sns.barplot(x = 'total_clicks', y = "publisher", data = publisher_category.head(nombre_publisher_slider),hue='category', dodge = False)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=17)
    plt.xlabel("Nombre de clics total", fontsize=24)
    plt.ylabel("Editeur", fontsize=26)
    plt.title("Nombre total de clics par editeur", fontsize=28)
    plt.show()
publisher_car_plot = publi_cat()
st.pyplot(publisher_car_plot)




st.header("Clics par éditeurs")
pub_df = pd.read_csv("reports_data.csv")
pub_grp_sr = pub_df.groupby(["publisher"])

pub_grp_df = pd.DataFrame(pub_grp_sr['unique_clicks'].sum().sort_values(ascending=False))
pub_grp_sum_df = pub_grp_df
pub_grp_sum_df.columns = ['uniq_tt']

pub_grp_mean_df = pd.DataFrame(pub_grp_sr['unique_clicks'].mean().sort_values(ascending=False))
pub_grp_mean_df.columns = ['uniq_moy']


pub_grp_merge_df = pub_grp_sum_df.join(pub_grp_mean_df)
pub_grp_merge_df.uniq_moy = pub_grp_merge_df.uniq_moy.round(2)


st.write('Nombre de clics pour les éditeurs les plus cliqués')
nombre_utilisateurs_unique_slide = st.slider("Nombre de journaux à représenter", 0,100,50)
pub_grp_merge_df_small50 = pub_grp_merge_df.head(nombre_utilisateurs_unique_slide )

def clics_editeurs():
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 11))
    # Plot the total clicks
    sns.set_color_codes("pastel")
    sns.barplot(x="uniq_tt", y=pub_grp_merge_df_small50.index, data=pub_grp_merge_df_small50,
                label="Total", color="g")
    # Plot the mean clicks
    sns.set_color_codes("colorblind")
    sns.barplot(x="uniq_moy", y=pub_grp_merge_df_small50.index, data=pub_grp_merge_df_small50,label="Moyen", color="g")
    # Add a legend and informate axis label
    ax.legend(ncol=1, loc="center right", frameon=True, fontsize=16, shadow=2)
    ax.set_xlabel("Nombre de clics")
    sns.despine(left=True, bottom=True)
    plt.title("Nombre de clics (moyen, total) par éditeur", fontdict={'fontsize': 18})
    plt.tight_layout()
st.pyplot(clicsEdi = clics_editeurs())



st.header("Nombre de clics des liens de la NewsLetter en fonction de leur position dans la newsletter")
classement = pd.read_csv("rank_data.csv")
classement_groupe = classement.groupby(by = 'Ranking')
clasement_groupe_data = pd.DataFrame(classement_groupe['Unique_clicks'].sum().sort_values(ascending=False))

st.write("Graphe des rangs:")
def rank_graphe():
    plt.figure(figsize=(14, 10))
    ax = sns.barplot("Unique_clicks", clasement_groupe_data.index, data=clasement_groupe_data, orient='h', color = 'firebrick')
    # ax.tick_params(axis='y', which='major', pad=20,length=20)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=17)
    plt.xlabel("Nombre de clics uniques", fontsize=24)
    plt.ylabel("Positionnement dans la Newsletter", fontsize=26)
    plt.title("Nombre total de clics uniques par position dans la newsletter", fontsize=28)
    # plt.tight_layout()
    # plt.savefig("nb_clics_rank_v3.png", dpi=200)
    plt.show()
graphe_rank = rank_graphe()
st.pyplot(graphe_rank)
#on travail maintenant sur les abonnés

st.title(" Analyse de l'audience")

st.header("Analyses des abonnés aux newsletter")
st.write('abonnés DeepNews')

deep_users = pd.read_csv("deep_users.csv")
if st.checkbox("voir la table des abonnés Deepnews"):
    st.write(deep_users)

Percentage_deep_users= deep_users.groupby(deep_users['MEMBER_RATING']).size()/len(deep_users['MEMBER_RATING'])*100
Rating_proportion_deep_users = pd.DataFrame({"Percentage": Percentage_deep_users})
Rating_proportion_deep_users.drop([1.0], inplace=True)


def piechart_deepnews():
    ax, fig= plt.subplots(figsize=(9,9))
    labels=["Aucun","Faible","Modéré","Elevé"]
    colors= ["#00A876", "#FF5900", "#0ACF00", "#FD0006"]
    plt.pie(Rating_proportion_deep_users['Percentage'], autopct='%1.0f%%', textprops={'fontsize': 22}, explode = (0, 0, 0, .05), colors=colors)
    plt.legend(labels, loc='best', bbox_to_anchor=(.9,.9), fontsize=18)
    plt.title("Engagement des abonnés Deepnews", fontsize=20, y=.95 )
    plt.tight_layout()
    # plt.savefig("rating_subscribers.png")
    # files.download("rating_subscribers.png")
    plt.show()



st.write('abonnés MondayNote')

monday_users = pd.read_csv('monday_users.csv')

if st.checkbox("Table des abonnés de la MondayNote"):
    st.write(monday_users)

st.write("Quelle est la proportion d'abonnés inscrits sur les deux newsletter ?")

Percentage_monday_users = monday_users.groupby(monday_users['MEMBER_RATING']).size()/len(monday_users['MEMBER_RATING'])*100
Rating_proportion_monday_users = pd.DataFrame({"Percentage": Percentage_monday_users})
Rating_proportion_monday_users.drop([1.0], inplace=True)


def piechart_mondaynote():
    ax, fig= plt.subplots(figsize=(9,9))
    labels=["Aucun","Faible","Modéré","Elevé"]
    colors= ["#00A876", "#FF5900", "#0ACF00", "#FD0006"]
    plt.pie(Rating_proportion_monday_users['Percentage'], autopct='%1.0f%%', textprops={'fontsize': 22}, explode = (0, 0, 0, .05), colors=colors)
    plt.legend(labels, loc='best', bbox_to_anchor=(.9,.9), fontsize=18)
    plt.title("Engagement des abonnés Monday Note", fontsize=20, y=.95 )
    plt.tight_layout()
    # plt.savefig("rating_subscribers.png")
    # files.download("rating_subscribers.png")
    plt.show()

monday_user = monday_users[["LEID","EUID","MEMBER_RATING"]]
deep_user = deep_users[["LEID","EUID","MEMBER_RATING"]]

joint = pd.merge(monday_user,deep_user, on = "EUID")
st.success(len(joint))
joint.rename(columns={"MEMBER_RATING_x": "Score MN", "LEID_x": "LEID MN", "MEMBER_RATING_y": "Score DN", "LEID_y": "LEID DN"}, inplace=True)
moy_joint=joint["Score DN"].mean()

others_MN = monday_user[~monday_user.EUID.isin(joint.EUID)].dropna()
moy_MN=others_MN["MEMBER_RATING"].mean()

moy_deep=deep_user["MEMBER_RATING"].mean()
others=deep_user[~deep_user.EUID.isin(joint.EUID)].dropna()

moy_others = others["MEMBER_RATING"].mean()
moy=[moy_joint, moy_others, moy_MN]

deepnews_part=len(joint)/len(deep_user.EUID)
monday_part=1-deepnews_part

def piechart_deep_vs_monday():

    fig, ax=plt.subplots()
    colors=["#ED002F", "#00A779"]
    labels=["Abonnés à la Monday Note", "Non abonnés à la Monday Note"]
    plt.pie([deepnews_part, monday_part], explode = [0,0.1], autopct='%1.0f%%', colors=colors, radius = 2, textprops={'fontsize': 22})
    plt.legend(labels, bbox_to_anchor=(-0.2,.6, 0.5, -0.1), fontsize=18)
    plt.title("Abonnés à la DeepNewsletter", y=1.8, fontsize=25)
    plt.tight_layout()
    # plt.savefig("Abonnés NL MN")
    # files.download("Abonnés NL MN.png")
    plt.show()

st.write("Voir le pie chart des abonnés vs non abonnés à la monday note")
pie_dvm = piechart_deep_vs_monday()
st.pyplot(pie_dvm)








piechart_selection = st.selectbox("Quelle représentation ?:",("Engagement DeepNews","Engagement MondayNote"))

if piechart_selection == "Engagement DeepNews":
    pc_deep = piechart_deepnews()
    st.pyplot(pc_deep)
else:
    pc_monday = piechart_mondaynote()
    st.pyplot(pc_monday)


def engagement_deux_NL():
    fig, ax=plt.subplots(figsize=(8,8))

    objects = ('Aux 2 NL', 'A la NL '+ '\n'+ 'Deepnews '+ '\n'+ 'uniquement', 'A la '+ '\n'+ 'Monday Note'+ '\n'+ 'uniquement')
    height= [moy_joint, moy_deep, moy_MN]
    y_pos = np.arange(len(objects))
    performance = list(moy)

    plt.bar(y_pos, height, align='center', alpha=.9, color=["#FF1800", "#00C12B", "#4013AF"])

    plt.ylabel('Engagement moyen', fontsize=20)
    plt.title('Engagement des abonnés', fontsize=25)

    plt.xticks(y_pos, objects, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
st.write("Voir l'engagement moyen des abonnés en fonction de la newsletter")
engagement = engagement_deux_NL()
st.pyplot(engagement)
if st.checkbox('voir commentaires'):
    st.write("L'engaement moyen est plus important pour DeepNews que pour la mondaynote")

st.header("Représentation des clics utilisateurs en fonction du thème de la newsletter")

theme = pd.read_csv('Digest Topic.csv')
theme.rename(columns={'Digest number':'digest'}, inplace=True)
clics = pd.read_csv('User_Clicks_Digest.csv')
clicks_theme = pd.merge(clics, theme, how = 'right',  on='digest' )
clicks_theme= clicks_theme[["EUID",'Thème',"Clicks"]]
clicks_theme_grouped = clicks_theme.groupby(by = ['EUID','Thème']).sum().sort_values('Clicks', ascending = False).reset_index()


nombre_utilisateurs_slide = st.slider("nombre d'utilisateurs à représenter",0,100,50)
clicks_theme_grouped_slide = clicks_theme_grouped.head(nombre_utilisateurs_slide)

def clics_par_theme():
    fig, ax = plt.subplots(figsize = ( 20,10 ))
    sns.barplot(x="EUID", y='Clicks',hue="Thème", data = clicks_theme_grouped_slide, dodge = False)
    plt.legend(loc = 'upper center', fontsize = 18)
    plt.xticks(rotation = 45)
    plt.xlabel('Identifiant uttilisateur', fontsize=18)
    plt.ylabel('Nombre de clics utilisateur', fontsize=18)

clics_theme_uti = clics_par_theme()
st.pyplot(clics_theme_uti)
