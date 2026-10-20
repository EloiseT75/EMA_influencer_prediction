import os
import numpy as np
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm
import pickle
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

print("="*60)
print("FEATURE ENGINEERING OPTIMISÉ V3 - ADVANCED")
print("="*60)

# =========================
# 1. CHARGEMENT DES DONNÉES
# =========================
print("\n Chargement des fichiers JSONL...")
train_data = pd.read_json('train.jsonl', lines=True)
train_data = json_normalize(train_data.to_dict(orient='records'))
print(f"✓ Train data: {len(train_data)} lignes")

kaggle_data = pd.read_json('kaggle_test.jsonl', lines=True)
kaggle_data = json_normalize(kaggle_data.to_dict(orient='records'))
print(f"✓ Test data: {len(kaggle_data)} lignes")

# =========================
# 2. EXTRACTION DU TEXTE COMPLET
# =========================
print("\n Extraction du texte complet...")
def extract_full_text(row):
    if 'extended_tweet.full_text' in row.index and pd.notna(row['extended_tweet.full_text']):
        return str(row['extended_tweet.full_text'])
    return str(row.get('text', ''))

train_data['full_text'] = train_data.apply(extract_full_text, axis=1)
kaggle_data['full_text'] = kaggle_data.apply(extract_full_text, axis=1)

# Nettoyage du texte pour TF-IDF
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    return text

train_data['clean_text'] = train_data['full_text'].apply(clean_text)
kaggle_data['clean_text'] = kaggle_data['full_text'].apply(clean_text)
print("✓ Texte extrait et nettoyé")

# =========================
# 3. FEATURES UTILISATEUR AVANCÉES
# =========================
print("\n Extraction des features utilisateur avancées...")

def extract_user_features(df):
    features = pd.DataFrame(index=df.index)
    
    def safe_get(col, default=0):
        if col in df.columns:
            return df[col].fillna(default)
        return pd.Series(default, index=df.index)
    
    # Features de base
    features['followers'] = safe_get('user.followers_count')
    features['friends'] = safe_get('user.friends_count')
    features['statuses'] = safe_get('user.statuses_count')
    features['favourites'] = safe_get('user.favourites_count')
    features['listed'] = safe_get('user.listed_count')
    
    # ==========================================
    # NOUVELLES FEATURES CRITIQUES POUR INFLUENCERS
    # ==========================================
    
    # Network asymmetry - CLÉ pour distinguer influencers
    features['network_asymmetry'] = np.abs(
        np.log1p(features['followers']) - np.log1p(features['friends'])
    )
    
    # Dominance du follower dans le réseau
    features['follower_dominance'] = (
        features['followers'] / (features['followers'] + features['friends'] + 1)
    )
    
    # Ratios follower/following améliorés
    features['follower_friend_ratio'] = features['followers'] / (features['friends'] + 1)
    features['friend_follower_ratio'] = features['friends'] / (features['followers'] + 1)
    
    # Ratio extrême (influencers ont souvent >10)
    features['extreme_follower_ratio'] = (features['follower_friend_ratio'] > 10).astype(int)
    features['balanced_network'] = (
        (features['follower_friend_ratio'] > 0.5) & 
        (features['follower_friend_ratio'] < 2.0)
    ).astype(int)
    
    # Log transformations
    features['log_followers'] = np.log1p(features['followers'])
    features['log_friends'] = np.log1p(features['friends'])
    features['log_statuses'] = np.log1p(features['statuses'])
    features['log_favourites'] = np.log1p(features['favourites'])
    features['log_listed'] = np.log1p(features['listed'])
    
    # Ratios d'activité
    features['tweets_per_follower'] = features['statuses'] / (features['followers'] + 1)
    features['tweets_per_friend'] = features['statuses'] / (features['friends'] + 1)
    features['favorites_per_tweet'] = features['favourites'] / (features['statuses'] + 1)
    features['listed_per_follower'] = features['listed'] / (features['followers'] + 1)
    
    # Score d'influence composite amélioré
    features['influence_score'] = (
        np.log1p(features['followers']) * 
        features['follower_friend_ratio'] * 
        np.log1p(features['listed'] + 1) *
        (1 + features['network_asymmetry'])
    )
    
    # Authority score (followers * listed / friends)
    features['authority_score'] = (
        features['followers'] * (features['listed'] + 1) / (features['friends'] + 1)
    )
    
    # Informations de profil
    features['has_description'] = safe_get('user.description', '').apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0
    )
    features['has_url'] = safe_get('user.url', '').apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0
    )
    features['has_location'] = safe_get('user.location', '').apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0
    )
    features['verified'] = safe_get('user.verified', False).astype(int)
    features['default_profile'] = safe_get('user.default_profile', True).astype(int)
    features['default_profile_image'] = safe_get('user.default_profile_image', True).astype(int)
    
    # Profile completeness score
    features['profile_completeness'] = (
        features['has_description'] + features['has_url'] + 
        features['has_location'] + features['verified']
    )
    
    # Ancienneté et activité
    if 'user.created_at' in df.columns:
        try:
            created_dates = pd.to_datetime(df['user.created_at'], 
                                          format='%a %b %d %H:%M:%S %z %Y', 
                                          errors='coerce', utc=True)
            reference_date = pd.to_datetime('2021-03-17', utc=True)
            features['account_age_days'] = (reference_date - created_dates).dt.days.fillna(0)
            features['account_age_years'] = features['account_age_days'] / 365.25
            features['tweets_per_day'] = features['statuses'] / (features['account_age_days'] + 1)
            
            # Mature account (>2 ans)
            features['mature_account'] = (features['account_age_years'] > 2).astype(int)
        except:
            features['account_age_days'] = 0
            features['account_age_years'] = 0
            features['tweets_per_day'] = 0
            features['mature_account'] = 0
    else:
        features['account_age_days'] = 0
        features['account_age_years'] = 0
        features['tweets_per_day'] = 0
        features['mature_account'] = 0
    
    # Description analysis
    if 'user.description' in df.columns:
        features['description_length'] = df['user.description'].fillna('').str.len()
        features['description_has_emoji'] = df['user.description'].fillna('').apply(
            lambda x: 1 if any(ord(c) > 127000 for c in str(x)) else 0
        )
        features['description_word_count'] = df['user.description'].fillna('').str.split().str.len()
    else:
        features['description_length'] = 0
        features['description_has_emoji'] = 0
        features['description_word_count'] = 0
        
    return features

user_features_train = extract_user_features(train_data)
user_features_test = extract_user_features(kaggle_data)
print(f"✓ {len(user_features_train.columns)} features utilisateur créées")

# =========================
# 4. FEATURES TWEET AVANCÉES
# =========================
print("\n Extraction des features de tweet avancées...")

def extract_tweet_features(df):
    features = pd.DataFrame(index=df.index)
    
    # Engagement metrics
    features['retweets'] = df.get('retweet_count', 0).fillna(0)
    features['favorites'] = df.get('favorite_count', 0).fillna(0)
    features['replies']'] = df.get('reply_count', 0).fillna(0)
    features['quotes'] = df.get('quote_count', 0).fillna(0)
    
    # Log transformations
    features['log_retweets'] = np.log1p(features['retweets'])
    features['log_favorites'] = np.log1p(features['favorites'])
    features['log_replies'] = np.log1p(features['replies'])
    
    # Total engagement
    features['total_engagement'] = (
        features['retweets'] + features['favorites'] + 
        features['replies'] + features['quotes']
    )
    features['log_engagement'] = np.log1p(features['total_engagement'])
    
    # Engagement composition
    features['retweet_favorite_ratio'] = features['retweets'] / (features['favorites'] + 1)
    features['reply_engagement_ratio'] = features['replies'] / (features['total_engagement'] + 1)
    features['quote_engagement_ratio'] = features['quotes'] / (features['total_engagement'] + 1)
    
    # Viralité indicators
    features['high_retweet'] = (features['retweets'] > 10).astype(int)
    features['high_favorite'] = (features['favorites'] > 20).astype(int)
    features['viral_tweet'] = (features['total_engagement'] > 50).astype(int)
    
    # Type de tweet
    features['is_retweet'] = df.get('retweeted', False).fillna(False).astype(int)
    features['is_quote'] = df.get('is_quote_status', False).fillna(False).astype(int)
    features['is_reply'] = df['in_reply_to_status_id'].notna().astype(int)
    features['is_original'] = (
        (~features['is_retweet'].astype(bool)) & 
        (~features['is_reply'].astype(bool))
    ).astype(int)
    
    features['possibly_sensitive'] = df.get('possibly_sensitive', False).fillna(False).astype(int)
    
    return features

tweet_features_train = extract_tweet_features(train_data)
tweet_features_test = extract_tweet_features(kaggle_data)
print(f"✓ {len(tweet_features_train.columns)} features de tweet créées")

# =========================
# 5. FEATURES TEMPORELLES
# =========================
print("\n Extraction des features temporelles...")

def extract_temporal_features(df):
    features = pd.DataFrame(index=df.index)
    
    try:
        tweet_dates = pd.to_datetime(df['created_at'], 
                                     format='%a %b %d %H:%M:%S %z %Y', 
                                     errors='coerce')
        
        features['hour_of_day'] = tweet_dates.dt.hour.fillna(12)
        features['day_of_week'] = tweet_dates.dt.dayofweek.fillna(3)
        features['day_of_month'] = tweet_dates.dt.day.fillna(15)
        features['month'] = tweet_dates.dt.month.fillna(3)
        
        # Patterns temporels (influencers postent stratégiquement)
        features['is_weekend'] = (tweet_dates.dt.dayofweek >= 5).fillna(False).astype(int)
        features['is_business_hours'] = (
            (tweet_dates.dt.hour >= 9) & (tweet_dates.dt.hour <= 18)
        ).fillna(False).astype(int)
        features['is_morning'] = (
            (tweet_dates.dt.hour >= 6) & (tweet_dates.dt.hour < 12)
        ).fillna(False).astype(int)
        features['is_afternoon'] = (
            (tweet_dates.dt.hour >= 12) & (tweet_dates.dt.hour < 18)
        ).fillna(False).astype(int)
        features['is_evening'] = (
            (tweet_dates.dt.hour >= 18) & (tweet_dates.dt.hour < 23)
        ).fillna(False).astype(int)
        features['is_night'] = (
            (tweet_dates.dt.hour >= 23) | (tweet_dates.dt.hour < 6)
        ).fillna(False).astype(int)
        
        # Prime time (heures de forte audience)
        features['is_prime_time'] = (
            ((tweet_dates.dt.hour >= 8) & (tweet_dates.dt.hour <= 10)) |
            ((tweet_dates.dt.hour >= 12) & (tweet_dates.dt.hour <= 14)) |
            ((tweet_dates.dt.hour >= 18) & (tweet_dates.dt.hour <= 21))
        ).fillna(False).astype(int)
        
    except Exception as e:
        print(f"Erreur temporelle: {e}")
        # Fallback values
        for col in ['hour_of_day', 'day_of_week', 'day_of_month', 'month',
                    'is_weekend', 'is_business_hours', 'is_morning', 'is_afternoon',
                    'is_evening', 'is_night', 'is_prime_time']:
            features[col] = 0
    
    return features

temporal_features_train = extract_temporal_features(train_data)
temporal_features_test = extract_temporal_features(kaggle_data)
print(f"✓ {len(temporal_features_train.columns)} features temporelles créées")

# =========================
# 6. FEATURES TEXTUELLES AVANCÉES
# =========================
print("\n Extraction des features textuelles avancées...")

def extract_text_features(df, text_col='full_text'):
    features = pd.DataFrame(index=df.index)
    texts = df[text_col].fillna('')
    
    # Longueur et mots
    features['text_len'] = texts.str.len()
    features['word_count'] = texts.str.split().str.len()
    features['avg_word_len'] = features['text_len'] / (features['word_count'] + 1)
    features['char_per_word'] = features['text_len'] / (features['word_count'] + 1)
    
    # Longueur optimale (influencers utilisent souvent l'espace efficacement)
    features['optimal_length'] = (
        (features['text_len'] >= 100) & (features['text_len'] <= 200)
    ).astype(int)
    
    # Ponctuation
    features['exclamation'] = texts.str.count('!')
    features['question'] = texts.str.count('\?')
    features['dots'] = texts.str.count('\.')
    features['commas'] = texts.str.count(',')
    features['punctuation_total'] = (
        features['exclamation'] + features['question'] + 
        features['dots'] + features['commas']
    )
    features['punctuation_density'] = features['punctuation_total'] / (features['text_len'] + 1)
    
    # Majuscules et style
    features['uppercase_ratio'] = texts.apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    features['digit_count'] = texts.str.count('\d')
    features['space_count'] = texts.str.count(' ')
    
    # Style shouting (CAPS)
    features['is_shouting'] = (features['uppercase_ratio'] > 0.5).astype(int)
    
    # Emojis
    features['emoji_count'] = texts.apply(
        lambda x: len([c for c in x if ord(c) > 127000])
    )
    features['has_emoji'] = (features['emoji_count'] > 0).astype(int)
    features['emoji_density'] = features['emoji_count'] / (features['text_len'] + 1)
    
    # ==========================================
    # FEATURES LINGUISTIQUES AVANCÉES
    # ==========================================
    
    # Call to action patterns (influencers)
    features['has_cta'] = texts.str.contains(
        r'(?i)(cliquez|suivez|abonnez|découvrez|rejoignez|visitez|regardez|écoutez)',
        regex=True, na=False
    ).astype(int)
    
    # Promotional language
    features['has_promotional'] = texts.str.contains(
        r'(?i)(nouveau|lancement|exclusif|offre|gratuit|promo|sortie|disponible)',
        regex=True, na=False
    ).astype(int)
    
    # Professional tone
    features['has_professional'] = texts.str.contains(
        r'(?i)(annonce|communiqué|partenariat|collaboration|projet|travail)',
        regex=True, na=False
    ).astype(int)
    
    # Conversational tone (observers)
    features['has_conversational'] = texts.str.contains(
        r'(?i)(merci|salut|bonjour|bonsoir|comment|pourquoi|quoi|penses)',
        regex=True, na=False
    ).astype(int)
    
    # Opinion/debate markers (observers)
    features['has_opinion'] = texts.str.contains(
        r'(?i)(je pense|selon moi|à mon avis|je trouve|perso)',
        regex=True, na=False
    ).astype(int)
    
    # Extraction des entités
    for prefix in ['entities', 'extended_tweet.entities']:
        if f'{prefix}.hashtags' in df.columns:
            features['hashtags'] = df[f'{prefix}.hashtags'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        if f'{prefix}.user_mentions' in df.columns:
            features['mentions'] = df[f'{prefix}.user_mentions'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        if f'{prefix}.urls' in df.columns:
            features['urls'] = df[f'{prefix}.urls'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        if f'{prefix}.media' in df.columns:
            features['media'] = df[f'{prefix}.media'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
    
    # Remplir les NaN
    for col in ['hashtags', 'mentions', 'urls', 'media']:
        if col not in features.columns:
            features[col] = 0
        else:
            features[col] = features[col].fillna(0)
    
    # Features binaires
    features['has_hashtag'] = (features['hashtags'] > 0).astype(int)
    features['has_mention'] = (features['mentions'] > 0).astype(int)
    features['has_url'] = (features['urls'] > 0).astype(int)
    features['has_media'] = (features['media'] > 0).astype(int)
    
    # Multiple entities (influencers utilisent souvent plusieurs)
    features['multiple_hashtags'] = (features['hashtags'] > 1).astype(int)
    features['multiple_mentions'] = (features['mentions'] > 1).astype(int)
    features['multiple_urls'] = (features['urls'] > 1).astype(int)
    
    # Ratios
    features['hashtag_ratio'] = features['hashtags'] / (features['word_count'] + 1)
    features['mention_ratio'] = features['mentions'] / (features['word_count'] + 1)
    features['url_ratio'] = features['urls'] / (features['word_count'] + 1)
    
    # Densité du contenu
    features['content_density'] = (
        features['hashtags'] + features['mentions'] + 
        features['urls'] + features['media']
    ) / (features['word_count'] + 1)
    
    # Rich content (media + urls)
    features['rich_content'] = (features['has_media'] | features['has_url']).astype(int)
    
    # Langue
    features['lang_fr'] = (df.get('lang', '') == 'fr').astype(int)
    features['lang_en'] = (df.get('lang', '') == 'en').astype(int)
    features['lang_und'] = (df.get('lang', '') == 'und').astype(int)
    features['lang_other'] = (~df.get('lang', '').isin(['fr', 'en', 'und'])).astype(int)
    
    return features

text_features_train = extract_text_features(train_data)
text_features_test = extract_text_features(kaggle_data)
print(f"✓ {len(text_features_train.columns)} features textuelles créées")

# =========================
# 7. FEATURES TF-IDF AMÉLIORÉES
# =========================
print("\n Extraction des features TF-IDF améliorées...")

# TF-IDF avec plus de features
tfidf = TfidfVectorizer(
    max_features=200,       # Augmenté de 100 à 200
    min_df=3,              # Plus permissif
    max_df=0.8,            # Plus permissif
    ngram_range=(1, 3),    # Trigrams
    strip_accents='unicode',
    sublinear_tf=True,     # Log scaling
    use_idf=True
)

tfidf_train = tfidf.fit_transform(train_data['clean_text'])
tfidf_test = tfidf.transform(kaggle_data['clean_text'])

# SVD avec plus de composantes
svd = TruncatedSVD(n_components=100, random_state=42)  # 100 au lieu de 50
tfidf_train_reduced = svd.fit_transform(tfidf_train)
tfidf_test_reduced = svd.transform(tfidf_test)

tfidf_features_train = pd.DataFrame(
    tfidf_train_reduced,
    columns=[f'tfidf_{i}' for i in range(100)],
    index=train_data.index
)
tfidf_features_test = pd.DataFrame(
    tfidf_test_reduced,
    columns=[f'tfidf_{i}' for i in range(100)],
    index=kaggle_data.index
)
print(f"✓ 100 features TF-IDF créées")

# Sauvegarder les transformers
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('svd_transformer.pkl', 'wb') as f:
    pickle.dump(svd, f)

# =========================
# 8. FEATURES COMBINÉES AVANCÉES
# =========================
print("\n Création des features combinées avancées...")

def create_combined_features(user_feats, tweet_feats, text_feats):
    combined = pd.DataFrame(index=user_feats.index)
    
    # ==========================================
    # ENGAGEMENT NORMALISÉ (CRITIQUE)
    # ==========================================
    combined['engagement_per_follower'] = (
        tweet_feats['total_engagement'] / (user_feats['followers'] + 1)
    )
    combined['retweet_per_follower'] = (
        tweet_feats['retweets'] / (user_feats['followers'] + 1)
    )
    combined['favorite_per_follower'] = (
        tweet_feats['favorites'] / (user_feats['followers'] + 1)
    )
    
    # Reach amplification
    combined['reach_amplification'] = (
        user_feats['followers'] * tweet_feats['retweets']
    )
    combined['log_reach'] = np.log1p(combined['reach_amplification'])
    
    # Virality score amélioré
    combined['virality_score'] = (
        tweet_feats['log_retweets'] * user_feats['log_followers'] * 
        user_feats['network_asymmetry']
    )
    
    # Potential reach
    combined['reach_potential'] = (
        user_feats['followers'] * (1 + tweet_feats['retweets'] + tweet_feats['quotes'])
    )
    
    # ==========================================
    # INTERACTION PATTERNS
    # ==========================================
    combined['interaction_intensity'] = (
        (text_feats['mentions'] + tweet_feats['replies']) / 
        (text_feats['word_count'] + 1)
    )
    
    # Broadcasting vs conversation
    combined['broadcast_score'] = (
        user_feats['follower_friend_ratio'] * 
        (1 - combined['interaction_intensity']) *
        (1 + text_feats['has_url'] + text_feats['has_media'])
    )
    
    # Conversation score (observers)
    combined['conversation_score'] = (
        user_feats['friend_follower_ratio'] * 
        combined['interaction_intensity'] *
        (1 + tweet_feats['is_reply'])
    )
    
    # ==========================================
    # CONTENT QUALITY & STRATEGY
    # ==========================================
    combined['content_quality'] = (
        text_feats['word_count'] * 
        (1 + text_feats['urls'] + text_feats['media']) *
        (1 + text_feats['hashtags'])
    )
    
    # Professional content score
    combined['professional_score'] = (
        (user_feats['verified'] + user_feats['has_url'] + 
         user_feats['has_description'] + text_feats['has_professional']) / 4
    )
    
    # Promotional intensity (influencers)
    combined['promotional_intensity'] = (
        text_feats['has_cta'] + text_feats['has_promotional'] + 
        text_feats['has_url'] + text_feats['has_media']
    ) / 4
    
    # ==========================================
    # INFLUENCE INDICATORS
    # ==========================================
    combined['influence_indicator'] = (
        user_feats['influence_score'] * 
        (1 + tweet_feats['log_engagement']) *
        (1 + combined['professional_score'])
    )
    
    # Authority combined with content
    combined['authority_content'] = (
        user_feats['authority_score'] * combined['content_quality']
    )
    
    # Network effect
    combined['network_effect'] = (
        user_feats['network_asymmetry'] * 
        user_feats['log_followers'] *
        (1 + tweet_feats['viral_tweet'])
    )
    
    # ==========================================
    # BEHAVIOR PATTERNS
    # ==========================================
    combined['activity_intensity'] = (
        user_feats['tweets_per_day'] * user_feats['log_statuses']
    )
    
    # Engagement efficiency
    combined['engagement_efficiency'] = (
        tweet_feats['total_engagement'] / (user_feats['statuses'] + 1)
    )
    
    # Content strategy score
    combined['content_strategy'] = (
        text_feats['optimal_length'] * text_feats['rich_content'] * 
        (1 + text_feats['hashtags'] + text_feats['mentions'])
    )
    
    return combined

combined_features_train = create_combined_features(
    user_features_train, tweet_features_train, text_features_train
)
combined_features_test = create_combined_features(
    user_features_test, tweet_features_test, text_features_test
)
print(f"✓ {len(combined_features_train.columns)} features combinées créées")

# =========================
# 9. ASSEMBLAGE FINAL
# =========================
print("\n Assemblage du dataset final...")

X_train = pd.concat([
    user_features_train,
    tweet_features_train,
    temporal_features_train,
    text_features_train,
    tfidf_features_train,
    combined_features_train
], axis=1)

X_test = pd.concat([
    user_features_test,
    tweet_features_test,
    temporal_features_test,
    text_features_test,
    tfidf_features_test,
    combined_features_test
], axis=1)

# Nettoyage
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"✓ Features finales: {X_train.shape[1]} colonnes")
print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Test shape: {X_test.shape}")

y_train = train_data['label'].values

# =========================
# 10. ANALYSE ET SAUVEGARDE
# =========================
print("\n Analyse des features...")

correlations = X_train.corrwith(pd.Series(y_train, index=X_train.index)).abs().sort_values(ascending=False)
print("\n Top 30 features:")
print(correlations.head(30))

print("\n Statistiques des features:")
print(f"Features avec corrélation > 0.1: {(correlations > 0.1).sum()}")
print(f"Features avec corrélation > 0.2: {(correlations > 0.2).sum()}")
print(f"Features avec corrélation > 0.3: {(correlations > 0.3).sum()}")

# Sauvegarder les correlations
correlations.to_csv('feature_correlations.csv', header=['correlation'])
print("\n Corrélations sauvegardées: feature_correlations.csv")

# =========================
# 11. SAUVEGARDE FINALE
# =========================
print("\n Sauvegarde des datasets...")

X_train.to_csv('X_train_features.csv', index=False)
X_test.to_csv('X_test_features.csv', index=False)
pd.DataFrame({'label': y_train}).to_csv('y_train.csv', index=False)
pd.DataFrame({'challenge_id': kaggle_data['challenge_id']}).to_csv('test_ids.csv', index=False)

# Sauvegarder les textes pour analyse
train_data[['full_text', 'clean_text']].to_csv('train_texts.csv', index=False)
kaggle_data[['challenge_id', 'full_text', 'clean_text']].to_csv('test_texts.csv', index=False)

# Sauvegarder les noms de features
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X_train.columns), f)

# Sauvegarder un mapping des groupes de features
feature_groups = {
    'user': list(user_features_train.columns),
    'tweet': list(tweet_features_train.columns),
    'temporal': list(temporal_features_train.columns),
    'text': list(text_features_train.columns),
    'tfidf': list(tfidf_features_train.columns),
    'combined': list(combined_features_train.columns)
}

with open('feature_groups.pkl', 'wb') as f:
    pickle.dump(feature_groups, f)

print("\n Fichiers sauvegardés:")
print("   ✓ X_train_features.csv")
print("   ✓ X_test_features.csv")
print("   ✓ y_train.csv")
print("   ✓ test_ids.csv")
print("   ✓ train_texts.csv")
print("   ✓ test_texts.csv")
print("   ✓ feature_names.pkl")
print("   ✓ feature_groups.pkl")
print("   ✓ feature_correlations.csv")
print("   ✓ tfidf_vectorizer.pkl")
print("   ✓ svd_transformer.pkl")

# =========================
# 12. RÉSUMÉ FINAL
# =========================
print("\n" + "="*60)
print(" RÉSUMÉ DU PREPROCESSING V3")
print("="*60)

print(f"\n Données:")
print(f"   Train samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Total features: {X_train.shape[1]}")

print(f"\n Répartition des features par groupe:")
for group_name, group_features in feature_groups.items():
    print(f"   {group_name.capitalize()}: {len(group_features)} features")

print(f"\n Distribution des labels (train):")
print(f"   Classe 0 (Observer): {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
print(f"   Classe 1 (Influencer): {(y_train==1).sum():,} ({(y_train==1).mean()*100:.1f}%)")

print(f"\n Top 10 features les plus corrélées:")
for idx, (feat, corr) in enumerate(correlations.head(10).items(), 1):
    print(f"   {idx:2d}. {feat:40s} : {corr:.4f}")

print(f"\n Nouvelles features clés ajoutées:")
print("   ✓ network_asymmetry (asymétrie du réseau)")
print("   ✓ follower_dominance (dominance des followers)")
print("   ✓ authority_score (score d'autorité)")
print("   ✓ influence_indicator (indicateur d'influence)")
print("   ✓ Features temporelles (hour, day, prime_time)")
print("   ✓ Features linguistiques (CTA, promotional, conversational)")
print("   ✓ TF-IDF amélioré (200 features → 100 SVD)")

print(f"\n Améliorations par rapport à V2:")
print(f"   • +{X_train.shape[1] - 150} features supplémentaires")
print("   • Features réseau plus sophistiquées")
print("   • Analyse temporelle complète")
print("   • Détection de patterns linguistiques")
print("   • TF-IDF avec trigrams")

print("\n" + "="*60)
print(" PREPROCESSING V3 TERMINÉ AVEC SUCCÈS!")
print("="*60)

print("\n Prochaines étapes:")
print("   1. Vérifier les corrélations dans feature_correlations.csv")
print("   2. Lancer catboost_training_v3_optimized.py")
print("   3. Analyser l'importance des features après entraînement")
print("   4. Soumettre sur Kaggle")

print("\n" + "="*60)