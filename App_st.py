"""
Hyperparamètres : 
-N size des n_grams
-Stopwords
-Avec ou sans Stopwords
-Documents


Interactions : 
-Ajout de documents
-Ajout de corpus
-Filtre
-Compare
-Reset text

"""
import streamlit as st
import streamlit.components.v1 as components
from Corpus import Corpus
from Document import Document
from Comparateur import PairText
from typing import List, Tuple
from Global_stuff import Global_stuff
from time import sleep

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Versus",
    page_icon="VERSUSLOGO.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
if 'global_stuff' not in st.session_state :
    st.session_state.global_stuff = Global_stuff

#Front-end complètement assisté par LLM parce que parlons franchement je savais pas comment on fait ça

def main():
    """Fonction principale de l'application Streamlit"""
    # Titre principal
    st.title("Versus")
        
    
    st.markdown("---")
    
    # Initialisation du state
    if 'corpus' not in st.session_state:
        st.session_state.corpus = Corpus()
    if 'filtered_corpus' not in st.session_state:
        st.session_state.filtered_corpus = None
    if 'comparateurs' not in st.session_state:
        st.session_state.comparateurs = dict()
    
    # Sidebar pour les paramètres globaux
    with st.sidebar:
        st.image("VERSUSLOGO.png", width=150)
        st.header("⚙️ Paramètres globaux")
        
        # Stopwords personnalisés
        st.subheader("Stopwords (Mots vides)")
        custom_stopwords = st.text_area(
            "Stopwords personnalisés (un par ligne)",
            value="\n".join(st.session_state.global_stuff.STOPWORDS),
            height=100
        )
        
        if st.button("Mettre à jour les stopwords"):
            st.session_state.global_stuff.STOPWORDS = set(
                word.strip().lower() for word in custom_stopwords.split('\n') 
                if word.strip()
            )
            st.success("Stopwords mis à jour!")

    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs(["Gestion du Corpus", "Filtrage et Tri", "Comparaisons", "Guide"])
    
    with tab1:
        st.header("Gestion du Corpus")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Ajout de documents
            st.subheader("Ajouter des documents")
            
            # Upload de fichiers
            uploaded_files = st.file_uploader(
                "Choisir des fichiers .txt",
                type=['txt'],
                accept_multiple_files=True
            )
            
            if uploaded_files :
                if st.button("Ajouter les fichiers uploadés") :
                    for uploaded_file in uploaded_files :
                        doc = Document(name=uploaded_file.name, file_value=uploaded_file.getvalue())
                        res = st.session_state.corpus.add_doc(doc)
                        st.info(res)
                        sleep(0.5)
                    st.rerun()
            
            # Ajout manuel
            st.subheader("Ajouter un document manuellement")
            doc_name = st.text_input("Nom du document")
            doc_content = st.text_area("Contenu du document", height=150)
            
            if st.button("Ajouter le document") and doc_name and doc_content:
                doc = Document(name=doc_name, content=doc_content)
                res = st.session_state.corpus.add_doc(doc)
                st.info(res)
                sleep(0.5)
                st.rerun()
        
        with col2:
            # Informations sur le corpus
            st.subheader("Informations du corpus")
            st.metric("Nombre de documents", len(st.session_state.corpus))
            
            if len(st.session_state.corpus) > 0:
                total_words = sum(len(doc.text.words) for doc in st.session_state.corpus.documents)
                st.metric("Total de mots", total_words)
                total_sentences = sum(len(doc.text.sentences) for doc in st.session_state.corpus.documents)
                st.metric("Total de phrases", total_sentences)
        
        # Liste des documents
        if len(st.session_state.corpus) > 0:
            st.subheader("Documents dans le corpus")
            
            for i, doc in enumerate(st.session_state.corpus.documents):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{doc.name}** ({len(doc.text.words)} mots)")
                
                with col2:
                    if st.button("Voir", key=f"view_{i}"):
                        st.text_area(f"Contenu de {doc.name}", doc.text.origin_content[:500] + "..." if len(doc.text.origin_content) > 500 else doc.text.origin_content, height=100, key=f"content_{i}")
                
                with col3:
                    if st.button("Supprimer", key=f"delete_{i}"):
                        st.session_state.corpus = st.session_state.corpus - i
                        st.rerun()
    
    with tab2:
        st.header("Filtrage et tri du Corpus")
        
        if len(st.session_state.corpus) == 0:
            st.warning("Aucun document dans le corpus. Ajoutez des documents d'abord.")
        else:
            st.subheader("1. Filtrage")
            regex_pattern = st.text_input("Mots filtrant", placeholder="Exemple: 'HUGO'")
            
            if st.button("Appliquer le filtre") and regex_pattern:
                st.session_state.filtered_corpus = st.session_state.corpus.filter(regex_pattern)
                st.success(f"Filtre appliqué! {len(st.session_state.filtered_corpus)} document(s) correspondent. Corpus actuel mis à jour")
            
            if st.button("Réinitialiser le filtre"):
                st.session_state.filtered_corpus = None
                st.success("Filtre réinitialisé.")
            
            # Affichage du corpus actuel (filtré s'il existe, sinon l'original)
            current_corpus = st.session_state.filtered_corpus if st.session_state.filtered_corpus else st.session_state.corpus
            
            if len(current_corpus) > 0:
                st.subheader(f"Corpus actuel ({len(current_corpus)} documents)")
                for doc in current_corpus.documents:
                    st.write(f"- {doc.name}")

            # Sélection du document source pour le tri
            st.subheader("2. Tri")
            source_doc_name = st.selectbox(
                "Document source",
                current_corpus.get_documents_names(),
                key="source_doc_tri"
            )
            
            if source_doc_name:
                source_doc = current_corpus.get_doc_by_name(source_doc_name)
                
                # Option pour trier par similarité
                if st.button("Trier le corpus par similarité"):
                    index_doc = current_corpus.index(source_doc)
                    with st.spinner(text="Tri du corpus...", show_time=True) : 
                        sorted_corpus = current_corpus.compare(index_doc)

                    st.success("Corpus actuel trié par similarité!")

                    with st.container():
                        for i, doc in enumerate(sorted_corpus.documents):
                            st.write(f"{i+1}. {doc.name}")

                    #On actualise le corpus actuel : 
                    if st.session_state.filtered_corpus : 
                        st.session_state.filtered_corpus = sorted_corpus #Le corpus filtré trié
                        st.session_state.corpus = sorted_corpus + st.session_state.corpus #Le corpus filtré trié et les documents restants en bout de file
                    else  :
                        st.session_state.corpus = sorted_corpus 
                    
    
    with tab3:
        st.header("Comparaisons de Documents")
        
        current_corpus = st.session_state.filtered_corpus if st.session_state.filtered_corpus else st.session_state.corpus
        
        if len(current_corpus) < 2:
            st.warning("Il faut au moins 2 documents pour faire des comparaisons.")
        else:
            # Sélection du document source
            st.subheader("1. Sélectionner le document source")
            source_doc_name = st.selectbox(
                "Document source",
                current_corpus.get_documents_names(),
                key="source_doc_compare"
            )
            
            if source_doc_name:
                source_doc = current_corpus.get_doc_by_name(source_doc_name)
    
                # Sélection du document cible
                st.subheader("2. Sélectionner le document cible")
                available_targets = [name for name in current_corpus.get_documents_names()]
                
                target_doc_name = st.selectbox(
                    "Document cible",
                    available_targets,
                    key="target_doc"
                )
                
                if target_doc_name:
                    target_doc = current_corpus.get_doc_by_name(target_doc_name)
                    
                    # Créer un comparateur
                    comparison_key_neutral = f"{source_doc_name}_vs_{target_doc_name}"

                    i = 1
                    comparison_key = comparison_key_neutral + "_" + str(i)
                    while comparison_key in st.session_state.comparateurs : 
                        i+=1
                        comparison_key = comparison_key_neutral + "_" + str(i)
                        

                    if st.button("Créer une comparaison"):
                        st.session_state.comparateurs[comparison_key] = PairText(source_doc.text, target_doc.text)
                        st.success(f"Comparaison créée: {source_doc_name} vs {target_doc_name}")
                        st.rerun()
        
        # Affichage des comparaisons existantes
        if st.session_state.comparateurs:
            st.subheader("3. Comparaisons actives")
            
            # Onglets pour chaque comparaison
            comparison_tabs = st.tabs(list(st.session_state.comparateurs.keys()))
            
            for i, (comp_key, comparateur) in enumerate(st.session_state.comparateurs.items()):
                with comparison_tabs[i]:
                    st.subheader(f"Comparaison: {comp_key}")
                    
                    # Paramètres de comparaison
                    size_choice, threshold_choice , diff_checkbox, stopword_checkbox, delete_comp = st.columns(5)
                    
                    with size_choice:
                        n_size = st.slider("Taille des n-grams", 1, 10, 3, key=f"n_{comp_key}")
                    
                    with threshold_choice:
                        threshold = st.slider("Seuil de similarité", 0.8, 1.0, 0.9, 0.01, key=f"threshold_{comp_key}")
                    
                    with diff_checkbox:
                        show_diff = st.checkbox("Montrer les différences", key=f"diff_{comp_key}")

                    with stopword_checkbox : 
                        ignore_stopwords = st.checkbox("Supprimer les stopwords", key=f"stopwords{comp_key}") 

                    with delete_comp:
                        if st.button("Supprimer", key=f"remove_{comp_key}"):
                            del st.session_state.comparateurs[comp_key]
                            st.rerun()
                    
                    if st.button("Comparer", key=f"compare_{comp_key}"):
                        # Effectuer la comparaison
                        
                        if ignore_stopwords : 
                            with st.spinner("Suppression des mots vides", show_time=True) :
                                comparateur.remove_stopwords_texts()
                                print("LA")
                        else : 
                            comparateur.set_default_texts()
                        
                        #
                        if threshold > 0.99 : 
                            threshold = 0.99
                        
                        with st.spinner("Comparaison en cours...", show_time=True) :
                            matches = comparateur.compare_n_grams(n=n_size, score_threshold=threshold, diff=show_diff)
                        st.success(f"{len(matches)} correspondance(s) trouvée(s)!")
                        
                        navigation_html = create_navigation_interface(matches, comparateur, comp_key, show_diff)
                        with open("navigation_output.html", 'w', encoding="utf-8") as f :
                            f.write(navigation_html)
                        #print(navigation_html)
                        components.html(navigation_html, height=1000, scrolling=True)
                        #st.markdown(navigation_html, unsafe_allow_html=True)

    with tab4 : 
        st.write("GUIDE : ")
        st.write("Pour éviter tout bug : ne pas mettre de titre contenant : des caractères spéciaux, des guillemets, des apostrophes")

def highlight_text_with_navigation(text, text_id, similarities, differences = None) -> str:
    """Ajoute la mise en surbrillance HTML au texte"""
    similarities_coords = similarities
    if not similarities_coords and not differences:
        return f'<div id="{text_id}" style="height: 400px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; white-space: pre-wrap; font-family: monospace;">{text}</div>'

    starts = [c[0] for c in similarities_coords]
    ends = [c[1] for c in similarities_coords]

    #Plutôt que d'une liste des coordonnées, on veut le nombre de fois où chacune apparaît

    color_sim = st.session_state.global_stuff.COLORS['sim']
    color_diff = st.session_state.global_stuff.COLORS['diff']
    
    # Construire le texte avec les balises HTML
    result = ""
    last_pos = 0

    #Trier les listes differences et similarities dans l'ordre d'apparition

    for i, (start, end) in enumerate(similarities_coords):
        # Ajouter le texte avant le surlignage

        result += text[last_pos:start]

        section = '' #La section de texte similaire
        
        if differences : #Si on doit afficher les différences
            if len(differences[i])>0 : #S'il existe des différences à afficher
                last_pos_diff = start
                for diff_start, diff_end in differences[i] : 
                    section += text[last_pos_diff:diff_start] #On ajoute le texte similaire normal
                    section += f'<span style="color: {color_diff};">{text[diff_start:diff_end]}</span>' #Et le texte différé en couleur
                    last_pos_diff = diff_end #Puis on incrémente
                section += text[diff_end:end] #On termine ce qui reste du texte

        else : 
            section += text[start:end]

        segment_idx = i
        # Imbrique la section dans une span à id unique. Plusieurs spans si plusieurs endroits amènent ici.
        
        # Construction de span ids

        section = f'<span id="{text_id}_segment_{segment_idx}" style="background-color: {color_sim}; cursor: pointer; border: 2px solid transparent;" onclick="navigateToSegment({segment_idx}, `{text_id}`)" onmouseover="this.style.border=\'2px solid #ff6600\'" onmouseout="this.style.border=\'2px solid transparent\'">{section}</span>'
        result += section
         
        last_pos = end
    
    # Ajouter le reste du texte
    result += text[last_pos:]
    
    # Entourer dans un div scrollable avec ID
    return f'''
<div id="{text_id}" style="background-color : #EBF3FC; height: 400px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; white-space: pre-wrap; font-family: monospace; line-height: 1.5;">
    {result}
</div>
    '''

def create_navigation_interface(matches: list, comparateur: PairText, comp_key: str, show_diff: bool = False) -> str:
    """Crée l'interface de navigation complète avec JavaScript"""
    
    # Préparer les données pour JavaScript
    similarities1 = []
    similarities2 = []
    differences1 = []
    differences2 = []
    
    text1_id = comp_key+"_"+comparateur.text1.name
    text2_id = comp_key+"_"+comparateur.text2.name
    print(text1_id)
    if show_diff :
        for sim1, sim2, diff1, diff2 in matches:
            similarities1.append(sim1)
            similarities2.append(sim2)
            differences1.append(diff1)
            differences2.append(diff2)
    else  :
         for sim1, sim2 in matches : 
            similarities1.append(sim1)
            similarities2.append(sim2)
           
    # Générer le HTML pour les deux textes
    text1_html = highlight_text_with_navigation(
        text=comparateur.text1.origin_content, 
        text_id=text1_id,
        similarities=similarities1, 
        differences=differences1 if show_diff else None,
    )
    
    text2_html = highlight_text_with_navigation(
        text=comparateur.text2.origin_content, 
        text_id=text2_id,
        similarities=similarities2, 
        differences=differences2 if show_diff else None,
    )
    
    # JavaScript pour la navigation synchronisée
    javascript = f"""
<script>
let currentSegment = 0;
let totalSegments = {len(similarities1)};

function navigateToSegment(segmentIndex, sourceTextId) {{
    currentSegment = segmentIndex;
    
    // Identifier le texte cible
    let targetTextId = sourceTextId === `{text1_id}` ? `{text2_id}` : `{text1_id}`;
    
    // Trouver les éléments source et cible
    let sourceElement = document.getElementById(sourceTextId + '_segment_' + segmentIndex);
    let targetElement = document.getElementById(targetTextId + '_segment_' + segmentIndex);
    
    if (sourceElement && targetElement) {{
        // Mettre en évidence les segments actifs
        highlightActiveSegments(segmentIndex);
        
        // Faire défiler les deux textes
        scrollToElement(sourceTextId, sourceElement);
        scrollToElement(targetTextId, targetElement);
        
        // Mettre à jour l'indicateur de navigation
        updateNavigationIndicator();
    }}
}}

function scrollToElement(containerId, element) {{
    let container = document.getElementById(containerId);
    if (container && element) {{
        let containerRect = container.getBoundingClientRect();
        let elementRect = element.getBoundingClientRect();
        let relativeTop = element.offsetTop - container.offsetTop;
        container.scrollTop = relativeTop - container.clientHeight / 2;
    }}
}}

function highlightActiveSegments(segmentIndex) {{
    // Retirer les anciens surlignages actifs
    document.querySelectorAll('.active-segment').forEach(el => {{
        el.classList.remove('active-segment');
        el.style.border = '2px solid transparent';
    }});
    
    // Ajouter le nouveau surlignage
    let element1 = document.getElementById(`{text1_id}_segment_` + segmentIndex);
    let element2 = document.getElementById(`{text2_id}_segment_` + segmentIndex);
    
    if (element1) {{
        element1.classList.add('active-segment');
        element1.style.border = '3px solid #ff0000';
    }}
    if (element2) {{
        element2.classList.add('active-segment');
        element2.style.border = '3px solid #ff0000';
    }}
}}

function nextSegment() {{
    if (currentSegment < totalSegments - 1) {{
        navigateToSegment(currentSegment + 1, `{text1_id}`);
    }}
}}

function prevSegment() {{
    if (currentSegment > 0) {{
        navigateToSegment(currentSegment - 1, `{text1_id}`);
    }}
}}

function updateNavigationIndicator() {{
    let indicator = document.getElementById('nav-indicator');
    if (indicator) {{
        indicator.innerHTML = `Segment ${{currentSegment + 1}} / ${{totalSegments}}`;
    }}
}}

// Navigation au clavier
document.addEventListener('keydown', function(e) {{
    if (e.key === 'ArrowRight' || e.key === 'n') {{
        nextSegment();
    }} else if (e.key === 'ArrowLeft' || e.key === 'p') {{
        prevSegment();
    }}
}});

// Initialiser le premier segment
setTimeout(() => {{
    if (totalSegments > 0) {{
        navigateToSegment(0, `{text1_id}`);
    }}
}}, 100);
</script>
    """
    
    # Interface complète avec contrôles de navigation
    navigation_controls = f"""
<div style="text-align: center; margin: 10px 0; padding: 10px; background-color: #EBF3FC; border-radius: 5px;">
    <button onclick="prevSegment()" style="margin: 0 5px; padding: 5px 15px; background-color: black; color: white; border: none; border-radius: 3px; cursor: pointer;">← Précédent</button>
    <span id="nav-indicator" style="color : black; margin: 0 15px; font-weight: bold;">Segment 1 / {len(similarities1)}</span>
    <button onclick="nextSegment()" style="margin: 0 5px; padding: 5px 15px; background-color: black; color: white; border: none; border-radius: 3px; cursor: pointer;">Suivant →</button>
</div>
<div style="font-size: 20px; color: white; text-align: center; margin-bottom: 10px;">
    Cliquez sur un segment surligné pour naviguer | Utilisez les flèches du clavier ou N/P
</div>
    """
    
    # Layout en colonnes
    layout = f"""
<div style="display: flex; gap: 20px;">
    <div style="flex: 1;">
        <h4 style="text-align: center; margin-bottom: 10px; color:white;">{comparateur.text1.name}</h4>
        {text1_html}
    </div>
    <div style="flex: 1;">
        <h4 style="text-align: center; margin-bottom: 10px; color:white;">{comparateur.text2.name}</h4>
        {text2_html}
    </div>
</div>
    """
    
    # Légende
    legend = f"""
    <div style="margin-top: 15px; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
        <br>- <span style="background-color: {st.session_state.global_stuff.COLORS['sim']}; padding: 2px 4px;">Texte surligné</span>: Passages similaires (cliquez pour naviguer)<br>
        <br>- <span style="color: {st.session_state.global_stuff.COLORS['diff']}; padding: 2px 4px;">Texte surligné</span>: Différences (si activées)<br>
        <br>- <span style="border: 3px solid {st.session_state.global_stuff.COLORS['sim_border']}; padding: 2px 4px;">Bordure rouge</span>: Segment actuellement sélectionné<br>
    </div>
    """
    
    return """<div style="font-family: 'Source Sans Pro', sans-serif;">"""+navigation_controls + layout + legend + javascript+'</div>'

main()