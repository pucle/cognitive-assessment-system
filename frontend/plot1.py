import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import requests
import json
import psycopg2
import psycopg2.extras
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Thêm thư viện cho biểu đồ nâng cao
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from pandas.plotting import parallel_coordinates

# Thiết lập style cho matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AlzheimerAnalysisVisualizer:
    def __init__(self, db_path=None, api_url=None, config=None):
        """
        Khởi tạo class phân tích và trực quan hóa dữ liệu Alzheimer
        
        Args:
            db_path: Đường dẫn database SQLite
            api_url: URL API backend
            config: Cấu hình kết nối
        """
        self.data = None
        self.X = None
        self.y = None
        self.models_results = {}
        self.db_path = db_path or "cognitive_assessment.db"
        self.api_url = api_url or "http://localhost:5000/api"
        self.config = config or {}
        
        # Cố gắng load dữ liệu từ nhiều nguồn
        self.load_data_from_system()
    
    def load_data_from_database(self):
        """Lấy dữ liệu từ PostgreSQL database với nhiều nguồn"""
        try:
            print("🔌 Đang kết nối PostgreSQL database...")

            # Thử kết nối PostgreSQL trước
            if self._load_from_postgresql():
                return True

            # Nếu không có PostgreSQL, thử SQLite backup
            print("⚠️ Không thể kết nối PostgreSQL, thử SQLite...")
            return self._load_from_sqlite_backup()

        except Exception as e:
            print(f"❌ Lỗi kết nối database: {e}")
            return False

    def _load_from_postgresql(self):
        """Kết nối và tải dữ liệu từ PostgreSQL"""
        try:
            # Lấy thông tin kết nối từ environment hoặc config
            db_config = {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': os.getenv('DATABASE_PORT', '5432'),
                'database': os.getenv('DATABASE_NAME', 'cognitive_assessment'),
                'user': os.getenv('DATABASE_USER', 'postgres'),
                'password': os.getenv('DATABASE_PASSWORD', ''),
            }

            # Kiểm tra xem có DATABASE_URL không
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                conn = psycopg2.connect(database_url)
            else:
                conn = psycopg2.connect(**db_config)

            print("✅ Kết nối PostgreSQL thành công")

            # Query để lấy dữ liệu từ cognitiveAssessmentResults
            query_main = """
            SELECT
                car.id,
                car.session_id,
                car.user_id,
                car.user_info->>'name' as user_name,
                car.user_info->>'age' as age,
                car.user_info->>'gender' as gender,
                car.user_info->>'email' as email,
                car.started_at,
                car.completed_at,
                car.total_questions,
                car.answered_questions,
                car.completion_rate,
                car.memory_score,
                car.cognitive_score,
                car.final_mmse_score,
                car.overall_gpt_score,
                car.cognitive_analysis,
                car.audio_features,
                car.question_results,
                car.status,
                car.created_at
            FROM cognitive_assessment_results car
            WHERE car.status = 'completed'
            AND car.final_mmse_score IS NOT NULL
            ORDER BY car.created_at DESC
            """

            # Query từ community assessments
            query_community = """
            SELECT
                ca.id,
                ca.session_id,
                ca.name,
                ca.email,
                ca.age,
                ca.gender,
                ca.phone,
                ca.final_mmse,
                ca.overall_gpt_score,
                ca.results_json,
                ca.created_at,
                'community' as source
            FROM community_assessments ca
            WHERE ca.status = 'completed'
            AND ca.final_mmse IS NOT NULL
            ORDER BY ca.created_at DESC
            """

            # Lấy dữ liệu chính
            df_main = pd.read_sql_query(query_main, conn)

            # Lấy dữ liệu community
            df_community = pd.read_sql_query(query_community, conn)

            conn.close()

            # Kết hợp dữ liệu
            combined_data = self._combine_database_data(df_main, df_community)

            if len(combined_data) > 0:
                self.data = combined_data
                print(f"✅ Đã tải {len(self.data)} records từ PostgreSQL")
                self.preprocess_data()
                return True
            else:
                print("⚠️ Không có dữ liệu hợp lệ trong PostgreSQL")
                return False

        except Exception as e:
            print(f"❌ Lỗi kết nối PostgreSQL: {e}")
            return False

    def _load_from_sqlite_backup(self):
        """Kết nối SQLite backup nếu PostgreSQL không khả dụng"""
        try:
            print(f"🔌 Đang kết nối SQLite backup: {self.db_path}")

            conn = sqlite3.connect(self.db_path)

            # Query tương thích với cấu trúc cũ
            query = """
            SELECT
                id,
                session_id as session_id,
                user_id,
                user_name,
                age,
                gender,
                email,
                started_at,
                completed_at,
                total_questions,
                answered_questions,
                completion_rate,
                memory_score,
                cognitive_score,
                final_mmse_score as mmse_score,
                overall_gpt_score,
                cognitive_analysis,
                audio_features,
                status,
                created_at
            FROM cognitive_assessment_results
            WHERE status = 'completed'
            AND final_mmse_score IS NOT NULL
            ORDER BY created_at DESC
            """

            self.data = pd.read_sql_query(query, conn)
            conn.close()

            if len(self.data) > 0:
                print(f"✅ Đã tải {len(self.data)} records từ SQLite backup")
                self.preprocess_data()
                return True
            else:
                print("⚠️ Không có dữ liệu trong SQLite backup")
                return False

        except Exception as e:
            print(f"❌ Lỗi kết nối SQLite backup: {e}")
            return False

    def _combine_database_data(self, df_main, df_community):
        """Kết hợp dữ liệu từ nhiều nguồn"""
        try:
            # Chuẩn hóa df_main
            df_main_clean = self._normalize_main_assessment_data(df_main)

            # Chuẩn hóa df_community
            df_community_clean = self._normalize_community_data(df_community)

            # Kết hợp
            combined = pd.concat([df_main_clean, df_community_clean], ignore_index=True)

            # Loại bỏ duplicates nếu có
            combined = combined.drop_duplicates(subset=['session_id'], keep='first')

            return combined

        except Exception as e:
            print(f"❌ Lỗi kết hợp dữ liệu: {e}")
            return pd.DataFrame()

    def _normalize_main_assessment_data(self, df):
        """Chuẩn hóa dữ liệu từ bảng cognitive_assessment_results"""
        try:
            # Extract features từ JSON columns
            df = df.copy()

            # Extract từ cognitive_analysis
            if 'cognitive_analysis' in df.columns:
                df['cognitive_analysis_parsed'] = df['cognitive_analysis'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x else {}
                )

                # Extract linguistic features
                df['avg_words_per_sentence'] = df['cognitive_analysis_parsed'].apply(
                    lambda x: x.get('linguistic_features', {}).get('avg_words_per_sentence', np.nan)
                )
                df['lexical_diversity'] = df['cognitive_analysis_parsed'].apply(
                    lambda x: x.get('linguistic_features', {}).get('lexical_diversity', np.nan)
                )
                df['semantic_fluency'] = df['cognitive_analysis_parsed'].apply(
                    lambda x: x.get('linguistic_features', {}).get('semantic_fluency', np.nan)
                )

                # Extract acoustic features
                df['speech_rate'] = df['cognitive_analysis_parsed'].apply(
                    lambda x: x.get('acoustic_features', {}).get('speech_rate', np.nan)
                )
                df['mean_pause_length'] = df['cognitive_analysis_parsed'].apply(
                    lambda x: x.get('acoustic_features', {}).get('mean_pause_length', np.nan)
                )
                df['f0_mean'] = df['cognitive_analysis_parsed'].apply(
                    lambda x: x.get('acoustic_features', {}).get('f0_mean', np.nan)
                )
                df['jitter'] = df['cognitive_analysis_parsed'].apply(
                    lambda x: x.get('acoustic_features', {}).get('jitter', np.nan)
                )
                df['shimmer'] = df['cognitive_analysis_parsed'].apply(
                    lambda x: x.get('acoustic_features', {}).get('shimmer', np.nan)
                )

            # Extract từ audio_features
            if 'audio_features' in df.columns:
                df['audio_features_parsed'] = df['audio_features'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x else {}
                )

                # Thêm các đặc trưng âm học bổ sung
                df['f0_std'] = df['audio_features_parsed'].apply(
                    lambda x: x.get('f0_std', np.nan)
                )
                df['hnr'] = df['audio_features_parsed'].apply(
                    lambda x: x.get('hnr', np.nan)
                )
                df['spectral_centroid'] = df['audio_features_parsed'].apply(
                    lambda x: x.get('spectral_centroid', np.nan)
                )

            # Thêm cột source
            df['source'] = 'main_assessment'

            # Đảm bảo các cột số có kiểu đúng
            numeric_cols = ['age', 'final_mmse_score', 'memory_score', 'cognitive_score',
                           'overall_gpt_score', 'completion_rate']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            print(f"❌ Lỗi chuẩn hóa dữ liệu main: {e}")
            return df

    def _normalize_community_data(self, df):
        """Chuẩn hóa dữ liệu từ bảng community_assessments"""
        try:
            df = df.copy()

            # Đổi tên cột để đồng nhất
            rename_dict = {
                'final_mmse': 'final_mmse_score',
                'overall_gpt_score': 'overall_gpt_score',
                'name': 'user_name',
                'results_json': 'results_json'
            }
            df = df.rename(columns=rename_dict)

            # Extract features từ results_json nếu có
            if 'results_json' in df.columns:
                df['results_parsed'] = df['results_json'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x else {}
                )

                # Extract một số features cơ bản
                df['avg_words_per_sentence'] = df['results_parsed'].apply(
                    lambda x: x.get('avg_words_per_sentence', np.nan)
                )
                df['speech_rate'] = df['results_parsed'].apply(
                    lambda x: x.get('speech_rate', np.nan)
                )

            # Đảm bảo các cột số
            numeric_cols = ['age', 'final_mmse_score', 'overall_gpt_score']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            print(f"❌ Lỗi chuẩn hóa dữ liệu community: {e}")
            return df
    
    def load_data_from_api(self):
        """Lấy dữ liệu từ API backend với nhiều endpoints"""
        try:
            print(f"🌐 Đang gọi API backend: {self.api_url}")

            # Danh sách endpoints cần gọi
            endpoints = {
                'cognitive_assessment_results': '/api/cognitive-assessment-results',
                'community_assessments': '/api/community-assessments',
                'users': '/api/users',
                'training_samples': '/api/training-samples',
                'model_performance': '/api/model-performance',
                'feature_analysis': '/api/feature-analysis',
                'assessment_stats': '/api/assessment-statistics'
            }

            all_data = {}

            for name, endpoint in endpoints.items():
                try:
                    print(f"🔄 Đang gọi {name}...")
                    response = requests.get(f"{self.api_url}{endpoint}",
                                          timeout=60,
                                          headers={'Accept': 'application/json'})

                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, dict) and 'data' in data:
                            all_data[name] = data['data']
                        elif isinstance(data, list):
                            all_data[name] = data
                        else:
                            all_data[name] = data

                        record_count = len(all_data[name]) if isinstance(all_data[name], list) else 1
                        print(f"✅ Đã lấy {name}: {record_count} records")
                    else:
                        print(f"⚠️ Không thể lấy {name}: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    print(f"❌ Lỗi API {name}: {e}")
                except json.JSONDecodeError as e:
                    print(f"❌ Lỗi parse JSON {name}: {e}")

            if all_data:
                # Kết hợp dữ liệu từ các endpoint
                self._combine_api_data(all_data)
                return True
            else:
                print("⚠️ Không lấy được dữ liệu từ bất kỳ API endpoint nào")
                return False

        except Exception as e:
            print(f"❌ Lỗi kết nối API: {e}")
            return False

    def _combine_api_data(self, api_data):
        """Kết hợp dữ liệu từ nhiều API endpoint"""
        try:
            print("🔧 Đang kết hợp dữ liệu từ API...")

            # Bắt đầu với dữ liệu cognitive assessment results
            main_data = []

            # Xử lý cognitive assessment results
            if 'cognitive_assessment_results' in api_data:
                cog_results = api_data['cognitive_assessment_results']
                if isinstance(cog_results, list):
                    for result in cog_results:
                        record = self._extract_cognitive_result_features(result)
                        record['source'] = 'api_cognitive'
                        main_data.append(record)
                print(f"📊 Xử lý {len(main_data)} cognitive assessment results")

            # Xử lý community assessments
            if 'community_assessments' in api_data:
                comm_results = api_data['community_assessments']
                if isinstance(comm_results, list):
                    for result in comm_results:
                        record = self._extract_community_features(result)
                        record['source'] = 'api_community'
                        main_data.append(record)
                print(f"📊 Thêm {len(comm_results)} community assessments")

            # Xử lý training samples nếu có
            if 'training_samples' in api_data:
                train_samples = api_data['training_samples']
                if isinstance(train_samples, list):
                    for sample in train_samples:
                        record = self._extract_training_sample_features(sample)
                        record['source'] = 'api_training'
                        main_data.append(record)
                print(f"📊 Thêm {len(train_samples)} training samples")

            # Tạo DataFrame
            if main_data:
                self.data = pd.DataFrame(main_data)
                print(f"✅ Kết hợp thành công: {len(self.data)} records tổng cộng")

                # Chuẩn hóa dữ liệu
                self._normalize_api_data()
                self.preprocess_data()
            else:
                print("⚠️ Không có dữ liệu hợp lệ từ API")
                self.data = pd.DataFrame()

        except Exception as e:
            print(f"❌ Lỗi kết hợp dữ liệu API: {e}")
            self.data = pd.DataFrame()

    def _extract_cognitive_result_features(self, result):
        """Extract features từ cognitive assessment result"""
        try:
            record = {
                'session_id': result.get('sessionId', result.get('session_id')),
                'user_id': result.get('userId', result.get('user_id')),
                'user_name': result.get('userInfo', {}).get('name', result.get('user_name')),
                'age': result.get('userInfo', {}).get('age', result.get('age')),
                'gender': result.get('userInfo', {}).get('gender', result.get('gender')),
                'email': result.get('userInfo', {}).get('email', result.get('email')),
                'final_mmse_score': result.get('finalMmseScore', result.get('final_mmse_score')),
                'memory_score': result.get('memoryScore', result.get('memory_score')),
                'cognitive_score': result.get('cognitiveScore', result.get('cognitive_score')),
                'overall_gpt_score': result.get('overallGptScore', result.get('overall_gpt_score')),
                'completion_rate': result.get('completionRate', result.get('completion_rate')),
                'created_at': result.get('createdAt', result.get('created_at')),
                'status': result.get('status', 'completed')
            }

            # Extract từ cognitive analysis
            cognitive_analysis = result.get('cognitiveAnalysis', result.get('cognitive_analysis', {}))
            if isinstance(cognitive_analysis, str):
                try:
                    cognitive_analysis = json.loads(cognitive_analysis)
                except:
                    cognitive_analysis = {}

            if cognitive_analysis:
                # Linguistic features
                ling_features = cognitive_analysis.get('linguistic_features', {})
                record.update({
                    'avg_words_per_sentence': ling_features.get('avg_words_per_sentence'),
                    'sentence_length': ling_features.get('sentence_length'),
                    'lexical_diversity': ling_features.get('lexical_diversity'),
                    'semantic_fluency': ling_features.get('semantic_fluency'),
                    'word_finding_difficulty': ling_features.get('word_finding_difficulty'),
                    'pause_frequency': ling_features.get('pause_frequency'),
                    'hesitation_rate': ling_features.get('hesitation_rate')
                })

                # Acoustic features
                aco_features = cognitive_analysis.get('acoustic_features', {})
                record.update({
                    'speech_rate': aco_features.get('speech_rate'),
                    'mean_pause_length': aco_features.get('mean_pause_length'),
                    'pause_duration_variance': aco_features.get('pause_duration_variance'),
                    'f0_mean': aco_features.get('f0_mean'),
                    'f0_std': aco_features.get('f0_std'),
                    'f0_range': aco_features.get('f0_range'),
                    'jitter': aco_features.get('jitter'),
                    'shimmer': aco_features.get('shimmer'),
                    'hnr': aco_features.get('hnr'),
                    'spectral_centroid': aco_features.get('spectral_centroid'),
                    'spectral_rolloff': aco_features.get('spectral_rolloff')
                })

            return record

        except Exception as e:
            print(f"❌ Lỗi extract cognitive result: {e}")
            return {}

    def _extract_community_features(self, result):
        """Extract features từ community assessment"""
        try:
            record = {
                'session_id': result.get('sessionId', result.get('session_id')),
                'user_name': result.get('name'),
                'email': result.get('email'),
                'age': result.get('age'),
                'gender': result.get('gender'),
                'phone': result.get('phone'),
                'final_mmse_score': result.get('finalMmse', result.get('final_mmse')),
                'overall_gpt_score': result.get('overallGptScore', result.get('overall_gpt_score')),
                'created_at': result.get('createdAt', result.get('created_at')),
                'status': result.get('status', 'completed')
            }

            # Extract từ results_json
            results_json = result.get('resultsJson', result.get('results_json'))
            if isinstance(results_json, str):
                try:
                    results_data = json.loads(results_json)
                    # Extract basic features
                    record.update({
                        'avg_words_per_sentence': results_data.get('avg_words_per_sentence'),
                        'speech_rate': results_data.get('speech_rate'),
                        'lexical_diversity': results_data.get('lexical_diversity'),
                        'memory_score': results_data.get('memory_score')
                    })
                except:
                    pass

            return record

        except Exception as e:
            print(f"❌ Lỗi extract community result: {e}")
            return {}

    def _extract_training_sample_features(self, sample):
        """Extract features từ training sample"""
        try:
            record = {
                'session_id': sample.get('sessionId', sample.get('session_id')),
                'user_email': sample.get('userEmail', sample.get('user_email')),
                'user_name': sample.get('userName', sample.get('user_name')),
                'question_id': sample.get('questionId', sample.get('question_id')),
                'question_text': sample.get('questionText', sample.get('question_text')),
                'auto_transcript': sample.get('autoTranscript', sample.get('auto_transcript')),
                'manual_transcript': sample.get('manualTranscript', sample.get('manual_transcript')),
                'created_at': sample.get('createdAt', sample.get('created_at')),
                'source': 'training'
            }

            # Có thể extract thêm features từ transcripts nếu cần
            return record

        except Exception as e:
            print(f"❌ Lỗi extract training sample: {e}")
            return {}

    def _normalize_api_data(self):
        """Chuẩn hóa dữ liệu từ API"""
        try:
            if self.data.empty:
                return

            # Chuẩn hóa các cột số
            numeric_cols = [
                'age', 'final_mmse_score', 'memory_score', 'cognitive_score',
                'overall_gpt_score', 'completion_rate', 'avg_words_per_sentence',
                'sentence_length', 'lexical_diversity', 'semantic_fluency',
                'speech_rate', 'mean_pause_length', 'f0_mean', 'f0_std',
                'jitter', 'shimmer', 'hnr', 'spectral_centroid'
            ]

            for col in numeric_cols:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # Chuẩn hóa diagnosis dựa trên MMSE score
            if 'final_mmse_score' in self.data.columns:
                self.data['diagnosis'] = self.data['final_mmse_score'].apply(
                    lambda x: 'normal' if pd.notna(x) and x >= 24 else
                             'mild_impairment' if pd.notna(x) and x >= 18 else
                             'moderate_impairment' if pd.notna(x) and x >= 10 else
                             'severe_impairment' if pd.notna(x) else 'unknown'
                )

            print(f"✅ Đã chuẩn hóa {len(self.data)} records từ API")

        except Exception as e:
            print(f"❌ Lỗi chuẩn hóa dữ liệu API: {e}")
    
    def combine_api_data(self, api_data):
        """Kết hợp dữ liệu từ nhiều API endpoint"""
        try:
            # Giả sử cấu trúc dữ liệu từ API
            assessments = api_data.get('/assessments/complete', [])
            
            if not assessments:
                print("⚠️ Không có dữ liệu assessment từ API")
                return
            
            # Chuyển đổi thành DataFrame
            records = []
            for assessment in assessments:
                record = {
                    'user_id': assessment.get('user_id'),
                    'age': assessment.get('age'),
                    'gender': assessment.get('gender'),
                    'diagnosis': assessment.get('diagnosis'),
                    'mmse_score': assessment.get('mmse_score'),
                    'moca_score': assessment.get('moca_score'),
                }
                
                # Thêm đặc trưng ngôn ngữ
                linguistic = assessment.get('linguistic_features', {})
                record.update({
                    'avg_words_per_sentence': linguistic.get('avg_words_per_sentence'),
                    'sentence_length': linguistic.get('sentence_length'),
                    'lexical_diversity': linguistic.get('lexical_diversity'),
                    'semantic_fluency': linguistic.get('semantic_fluency'),
                    'word_finding_difficulty': linguistic.get('word_finding_difficulty'),
                })
                
                # Thêm đặc trưng âm học
                acoustic = assessment.get('acoustic_features', {})
                record.update({
                    'speech_rate': acoustic.get('speech_rate'),
                    'mean_pause_length': acoustic.get('mean_pause_length'),
                    'f0_mean': acoustic.get('f0_mean'),
                    'f0_std': acoustic.get('f0_std'),
                    'jitter': acoustic.get('jitter'),
                    'shimmer': acoustic.get('shimmer'),
                    'hnr': acoustic.get('hnr'),
                })
                
                records.append(record)
            
            self.data = pd.DataFrame(records)
            print(f"✅ Đã kết hợp dữ liệu API: {len(self.data)} records")
            self.preprocess_data()
            
        except Exception as e:
            print(f"❌ Lỗi kết hợp dữ liệu API: {e}")
    
    def load_data_from_csv(self, csv_path):
        """Lấy dữ liệu từ file CSV"""
        try:
            print(f"📁 Đang đọc file CSV: {csv_path}")
            self.data = pd.read_csv(csv_path)
            print(f"✅ Đã đọc {len(self.data)} records từ CSV")
            self.preprocess_data()
            return True
        except Exception as e:
            print(f"❌ Lỗi đọc CSV: {e}")
            return False
    
    def load_data_from_system(self):
        """Thử load dữ liệu từ nhiều nguồn theo thứ tự ưu tiên"""
        print("🔍 Đang tìm nguồn dữ liệu...")
        
        # 1. Thử database trước
        if self.load_data_from_database():
            return
        
        # 2. Thử API
        if self.load_data_from_api():
            return
        
        # 3. Thử các file CSV có thể có
        csv_files = [
            'data/alzheimer_data.csv',
            'data/assessment_data.csv',
            'cognitive_data.csv',
            'alzheimer_features.csv'
        ]
        
        for csv_file in csv_files:
            if self.load_data_from_csv(csv_file):
                return
        
        # 4. Cuối cùng mới tạo dữ liệu demo
        print("⚠️ Không tìm thấy dữ liệu thực, tạo dữ liệu demo...")
        self.generate_realistic_sample_data()
    
    def preprocess_data(self):
        """Tiền xử lý dữ liệu"""
        print("🔧 Đang tiền xử lý dữ liệu...")
        
        # Xử lý missing values
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())
        
        # Xử lý categorical data
        if 'diagnosis' in self.data.columns:
            # Chuẩn hóa nhãn diagnosis
            diagnosis_mapping = {
                'alzheimer': 1, 'ad': 1, 'dementia': 1,
                'normal': 0, 'control': 0, 'healthy': 0,
                'mci': 0.5  # Mild Cognitive Impairment
            }
            
            self.data['diagnosis_clean'] = self.data['diagnosis'].str.lower().map(diagnosis_mapping)
            self.data['diagnosis_clean'] = self.data['diagnosis_clean'].fillna(0)
            
            # Tạo nhãn binary (0: Normal/MCI, 1: Alzheimer)
            self.data['label'] = (self.data['diagnosis_clean'] == 1).astype(int)
            
            # Tạo nhóm cho visualization
            self.data['group'] = self.data['label'].map({0: 'Control', 1: 'Alzheimer'})
        
        # Xử lý outliers (loại bỏ các giá trị quá bất thường)
        for col in numeric_columns:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data[col] = self.data[col].clip(lower_bound, upper_bound)
        
        print(f"✅ Tiền xử lý hoàn tất. Dữ liệu: {self.data.shape}")
        print(f"📊 Phân bố nhãn: {self.data['group'].value_counts().to_dict()}")
    
    def generate_realistic_sample_data(self):
        """Tạo dữ liệu mẫu thực tế hơn với độ khó phân loại cao"""
        print("🔄 Tạo dữ liệu mẫu thực tế...")
        np.random.seed(42)
        
        n_samples = 300
        n_alzheimer = 150
        n_control = 150
        
        # Tạo dữ liệu với sự chồng chéo cao hơn để tránh overfitting
        data_records = []
        
        for i in range(n_samples):
            is_alzheimer = i < n_alzheimer
            
            if is_alzheimer:
                # Nhóm Alzheimer - có biến thiên cao
                record = {
                    'user_id': f'ALZ_{i:03d}',
                    'group': 'Alzheimer',
                    'label': 1,
                    'age': np.random.normal(78, 10),
                    'gender': np.random.choice(['M', 'F']),
                    'education_level': np.random.randint(8, 16),
                    'mmse_score': np.random.normal(19, 6),  # Có overlap với control
                    'moca_score': np.random.normal(17, 5),
                    
                    # Đặc trưng ngôn ngữ - có noise
                    'avg_words_per_sentence': np.random.normal(7.2, 3.0) + np.random.normal(0, 1),
                    'sentence_length': np.random.normal(48, 20) + np.random.normal(0, 5),
                    'stop_word_frequency': np.random.normal(0.32, 0.12) + np.random.normal(0, 0.02),
                    'lexical_diversity': np.random.normal(0.48, 0.15) + np.random.normal(0, 0.05),
                    'semantic_fluency': np.random.normal(13, 5) + np.random.normal(0, 2),
                    'word_finding_difficulty': np.random.normal(0.22, 0.10) + np.random.normal(0, 0.03),
                    
                    # Đặc trưng âm học - có noise
                    'speech_rate': np.random.normal(115, 30) + np.random.normal(0, 10),
                    'mean_pause_length': np.random.normal(0.75, 0.35) + np.random.normal(0, 0.1),
                    'pause_frequency': np.random.normal(0.14, 0.06) + np.random.normal(0, 0.02),
                    'f0_mean': np.random.normal(175, 40) + np.random.normal(0, 15),
                    'f0_std': np.random.normal(42, 15) + np.random.normal(0, 5),
                    'jitter': np.random.normal(0.011, 0.005) + np.random.normal(0, 0.002),
                    'shimmer': np.random.normal(0.032, 0.012) + np.random.normal(0, 0.003),
                    'hnr': np.random.normal(19, 5) + np.random.normal(0, 2),
                }
            else:
                # Nhóm Control - cũng có biến thiên
                record = {
                    'user_id': f'CTL_{i-n_alzheimer:03d}',
                    'group': 'Control',
                    'label': 0,
                    'age': np.random.normal(72, 12),
                    'gender': np.random.choice(['M', 'F']),
                    'education_level': np.random.randint(10, 18),
                    'mmse_score': np.random.normal(27, 3),  # Có overlap với Alzheimer
                    'moca_score': np.random.normal(25, 3),
                    
                    # Đặc trưng ngôn ngữ
                    'avg_words_per_sentence': np.random.normal(9.1, 2.5) + np.random.normal(0, 1),
                    'sentence_length': np.random.normal(68, 22) + np.random.normal(0, 5),
                    'stop_word_frequency': np.random.normal(0.26, 0.08) + np.random.normal(0, 0.02),
                    'lexical_diversity': np.random.normal(0.62, 0.12) + np.random.normal(0, 0.05),
                    'semantic_fluency': np.random.normal(19, 4) + np.random.normal(0, 2),
                    'word_finding_difficulty': np.random.normal(0.09, 0.05) + np.random.normal(0, 0.02),
                    
                    # Đặc trưng âm học
                    'speech_rate': np.random.normal(142, 25) + np.random.normal(0, 10),
                    'mean_pause_length': np.random.normal(0.48, 0.22) + np.random.normal(0, 0.08),
                    'pause_frequency': np.random.normal(0.07, 0.04) + np.random.normal(0, 0.015),
                    'f0_mean': np.random.normal(162, 35) + np.random.normal(0, 12),
                    'f0_std': np.random.normal(32, 12) + np.random.normal(0, 4),
                    'jitter': np.random.normal(0.007, 0.003) + np.random.normal(0, 0.001),
                    'shimmer': np.random.normal(0.023, 0.009) + np.random.normal(0, 0.002),
                    'hnr': np.random.normal(23, 4) + np.random.normal(0, 1.5),
                }
            
            data_records.append(record)
        
        self.data = pd.DataFrame(data_records)
        
        # Đảm bảo các giá trị trong phạm vi hợp lý
        self.data['mmse_score'] = self.data['mmse_score'].clip(0, 30)
        self.data['moca_score'] = self.data['moca_score'].clip(0, 30)
        self.data['lexical_diversity'] = self.data['lexical_diversity'].clip(0, 1)
        self.data['stop_word_frequency'] = self.data['stop_word_frequency'].clip(0, 1)
        self.data['jitter'] = self.data['jitter'].clip(0, 0.1)
        self.data['shimmer'] = self.data['shimmer'].clip(0, 0.2)
        
        print(f"✅ Đã tạo dữ liệu mẫu thực tế: {self.data.shape}")
        print(f"📊 Phân bố: {self.data['group'].value_counts().to_dict()}")
    
    def create_feature_comparison_table(self):
        """Tạo bảng so sánh đặc trưng với statistical test chi tiết"""
        print("\n📊 Tạo bảng so sánh đặc trưng chi tiết...")

        from scipy import stats

        # Danh sách đầy đủ các đặc trưng cần phân tích
        all_features = {
            # Đặc trưng ngôn ngữ
            'avg_words_per_sentence': 'Số từ trung bình/câu',
            'sentence_length': 'Độ dài câu (từ)',
            'lexical_diversity': 'Đa dạng từ vựng',
            'semantic_fluency': 'Trôi chảy ngữ nghĩa',
            'word_finding_difficulty': 'Khó khăn tìm từ',
            'pause_frequency': 'Tần suất dừng',
            'hesitation_rate': 'Tỷ lệ do dự',

            # Đặc trưng âm học cơ bản
            'speech_rate': 'Tốc độ nói (từ/phút)',
            'mean_pause_length': 'Độ dài dừng trung bình (giây)',
            'pause_duration_variance': 'Phương sai độ dài dừng',

            # Đặc trưng âm sắc (Prosody)
            'f0_mean': 'F0 trung bình (Hz)',
            'f0_std': 'Độ lệch chuẩn F0',
            'f0_range': 'Phạm vi F0',

            # Đặc trưng chất lượng giọng
            'jitter': 'Jitter (%)',
            'shimmer': 'Shimmer (%)',
            'hnr': 'HNR (dB)',

            # Đặc trưng phổ tần
            'spectral_centroid': 'Trọng tâm phổ',
            'spectral_rolloff': 'Spectral Rolloff',

            # Đặc trưng nhận thức
            'final_mmse_score': 'Điểm MMSE',
            'memory_score': 'Điểm trí nhớ',
            'cognitive_score': 'Điểm nhận thức tổng thể',
            'overall_gpt_score': 'Điểm GPT tổng thể',

            # Thông tin nhân khẩu học
            'age': 'Tuổi',
            'completion_rate': 'Tỷ lệ hoàn thành (%)'
        }

        # Lọc các features có trong dữ liệu và có đủ dữ liệu để phân tích
        available_features = {}
        for feature, description in all_features.items():
            if feature in self.data.columns:
                # Kiểm tra có đủ dữ liệu cho phân tích không
                alzheimer_data = self.data[self.data['group'] == 'Alzheimer'][feature].dropna()
                control_data = self.data[self.data['group'] == 'Control'][feature].dropna()

                if len(alzheimer_data) >= 3 and len(control_data) >= 3:  # Cần ít nhất 3 mẫu
                    available_features[feature] = description

        print(f"📊 Phân tích {len(available_features)} đặc trưng...")

        comparison_stats = []

        for feature, description in available_features.items():
            try:
                alzheimer_data = self.data[self.data['group'] == 'Alzheimer'][feature].dropna()
                control_data = self.data[self.data['group'] == 'Control'][feature].dropna()

                if len(alzheimer_data) > 0 and len(control_data) > 0:
                    # Thực hiện t-test (two-tailed)
                    t_stat, p_value = stats.ttest_ind(alzheimer_data, control_data, equal_var=False)

                    # Tính effect size (Cohen's d) với hiệu chỉnh cho sample size nhỏ
                    pooled_std = np.sqrt(((len(alzheimer_data)-1)*alzheimer_data.std()**2 +
                                        (len(control_data)-1)*control_data.std()**2) /
                                       (len(alzheimer_data) + len(control_data) - 2))

                    if pooled_std > 0:
                        cohens_d = abs(alzheimer_data.mean() - control_data.mean()) / pooled_std
                    else:
                        cohens_d = 0

                    # Tính confidence interval cho mean difference
                    mean_diff = alzheimer_data.mean() - control_data.mean()
                    se_diff = np.sqrt(alzheimer_data.var()/len(alzheimer_data) + control_data.var()/len(control_data))
                    ci_lower = mean_diff - 1.96 * se_diff
                    ci_upper = mean_diff + 1.96 * se_diff

                    # Phân loại effect size
                    if cohens_d < 0.2:
                        effect_size_category = "Small"
                    elif cohens_d < 0.5:
                        effect_size_category = "Medium"
                    elif cohens_d < 0.8:
                        effect_size_category = "Large"
                    else:
                        effect_size_category = "Very Large"

                    stats_row = {
                        'Feature': description,
                        'Feature_Code': feature,
                        'Alzheimer_N': len(alzheimer_data),
                        'Control_N': len(control_data),
                        'Alzheimer_Mean': f"{alzheimer_data.mean():.3f}",
                        'Alzheimer_Std': f"{alzheimer_data.std():.3f}",
                        'Control_Mean': f"{control_data.mean():.3f}",
                        'Control_Std': f"{control_data.std():.3f}",
                        'Mean_Diff': f"{mean_diff:.3f}",
                        'CI_95%': f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                        'Cohens_D': f"{cohens_d:.3f}",
                        'Effect_Size': effect_size_category,
                        'T_Statistic': f"{t_stat:.3f}",
                        'P_Value': f"{p_value:.6f}",
                        'Significant': "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "NS"
                    }
                    comparison_stats.append(stats_row)

            except Exception as e:
                print(f"⚠️ Lỗi phân tích feature {feature}: {e}")
                continue

        if not comparison_stats:
            print("⚠️ Không có đủ dữ liệu để tạo bảng so sánh")
            return pd.DataFrame()

        comparison_df = pd.DataFrame(comparison_stats)

        # Sắp xếp theo effect size (từ lớn đến nhỏ)
        comparison_df = comparison_df.sort_values('Cohens_D', ascending=False)

        # In bảng với format đẹp
        print("\n" + "="*150)
        print("BẢNG SO SÁNH ĐẶC TRƯNG NGÔN NGỮ VÀ ÂM HỌC")
        print("="*150)
        print(comparison_df[['Feature', 'Alzheimer_N', 'Control_N', 'Alzheimer_Mean', 'Control_Mean',
                           'Mean_Diff', 'Cohens_D', 'Effect_Size', 'Significant']].to_string(index=False))

        # Tóm tắt kết quả
        print(f"\n📊 Tóm tắt:")
        print(f"   • Tổng số đặc trưng phân tích: {len(comparison_df)}")
        print(f"   • Đặc trưng có ý nghĩa thống kê (p < 0.05): {len(comparison_df[comparison_df['Significant'] != 'NS'])}")
        print(f"   • Đặc trưng có effect size lớn (Cohen's d ≥ 0.5): {len(comparison_df[comparison_df['Cohens_D'].astype(float) >= 0.5])}")

        # Phân tích theo loại đặc trưng
        linguistic_features = comparison_df[comparison_df['Feature_Code'].isin([
            'avg_words_per_sentence', 'sentence_length', 'lexical_diversity',
            'semantic_fluency', 'word_finding_difficulty', 'pause_frequency', 'hesitation_rate'
        ])]

        acoustic_features = comparison_df[comparison_df['Feature_Code'].isin([
            'speech_rate', 'mean_pause_length', 'pause_duration_variance',
            'f0_mean', 'f0_std', 'f0_range', 'jitter', 'shimmer', 'hnr',
            'spectral_centroid', 'spectral_rolloff'
        ])]

        cognitive_features = comparison_df[comparison_df['Feature_Code'].isin([
            'final_mmse_score', 'memory_score', 'cognitive_score', 'overall_gpt_score'
        ])]

        print("🔤 Đặc trưng ngôn ngữ có ý nghĩa: {}/{}".format(
            len(linguistic_features[linguistic_features['Significant'] != 'NS']),
            len(linguistic_features)
        ))
        print("🔊 Đặc trưng âm học có ý nghĩa: {}/{}".format(
            len(acoustic_features[acoustic_features['Significant'] != 'NS']),
            len(acoustic_features)
        ))
        print("🧠 Đặc trưng nhận thức có ý nghĩa: {}/{}".format(
            len(cognitive_features[cognitive_features['Significant'] != 'NS']),
            len(cognitive_features)
        ))

        return comparison_df
    
    def train_models_properly(self):
        """Huấn luyện các mô hình với cross-validation và tránh overfitting"""
        print("\n🤖 Huấn luyện các mô hình với cross-validation...")
        
        # Chuẩn bị dữ liệu
        feature_columns = [col for col in self.data.columns 
                          if col not in ['group', 'label', 'user_id', 'gender', 'diagnosis', 'diagnosis_clean']]
        
        # Chỉ lấy các cột số
        numeric_features = []
        for col in feature_columns:
            if self.data[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
        
        X = self.data[numeric_features].fillna(self.data[numeric_features].median())
        y = self.data['label']
        
        print(f"📊 Features được sử dụng: {len(numeric_features)}")
        print(f"📊 Số samples: {len(X)}")
        
        # Chia dữ liệu với stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Định nghĩa các mô hình với hyperparameters thực tế
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_split=5,
                min_samples_leaf=3, random_state=42, class_weight='balanced'
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', probability=True, random_state=42,
                gamma='scale', class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                C=0.5, random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'MLP Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25), max_iter=1000,
                alpha=0.001, random_state=42, early_stopping=True,
                validation_fraction=0.2, learning_rate='adaptive'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=4,
                min_samples_split=5, min_samples_leaf=3, random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, learning_rate=0.5, random_state=42
            ),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=6, min_samples_split=10, min_samples_leaf=5,
                random_state=42, class_weight='balanced'
            )
        }

        # Thử thêm XGBoost nếu có
        try:
            from xgboost import XGBClassifier
            models['XGBoost'] = XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=4,
                min_child_weight=3, random_state=42, scale_pos_weight=len(y)/sum(y)
            )
        except ImportError:
            print("⚠️ XGBoost không khả dụng, bỏ qua")

        # Thử thêm LightGBM nếu có
        try:
            from lightgbm import LGBMClassifier
            models['LightGBM'] = LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=4,
                min_child_samples=10, random_state=42, class_weight='balanced'
            )
        except ImportError:
            print("⚠️ LightGBM không khả dụng, bỏ qua")

        # Thử thêm CatBoost nếu có
        try:
            from catboost import CatBoostClassifier
            models['CatBoost'] = CatBoostClassifier(
                iterations=100, learning_rate=0.1, depth=4,
                verbose=False, random_state=42, auto_class_weights='Balanced'
            )
        except ImportError:
            print("⚠️ CatBoost không khả dụng, bỏ qua")
        
        results = []
        
        for name, model in models.items():
            print(f"🔄 Đang huấn luyện {name}...")

            # Phân loại mô hình để xử lý khác nhau
            needs_scaling = name in ['SVM', 'Logistic Regression', 'MLP Neural Network', 'Naive Bayes']
            is_tree_based = name in ['Random Forest', 'Gradient Boosting', 'Decision Tree', 'XGBoost', 'LightGBM', 'CatBoost', 'AdaBoost']

            # Cross-validation với 5-fold stratified
            try:
                if needs_scaling:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                else:  # Tree-based models
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            except Exception as e:
                print(f"❌ Lỗi khi huấn luyện mô hình {name}: {e}")
                continue

            # Tính các metrics chi tiết
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Balanced Accuracy
            balanced_acc = (recall + specificity) / 2

            # ROC AUC và PRC AUC
            roc_auc = 0.5
            pr_auc = 0.5

            if y_prob is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)

                    # Precision-Recall AUC
                    from sklearn.metrics import precision_recall_curve, auc as auc_pr
                    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
                    pr_auc = auc_pr(recall_curve, precision_curve)
                except:
                    roc_auc = 0.5
                    pr_auc = 0.5

            # Matthews Correlation Coefficient
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(y_test, y_pred)

            # Cohen's Kappa
            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(y_test, y_pred)

            # Training time (approximate)
            training_time = "N/A"  # Có thể tính thời gian thực nếu cần

            results.append({
                'Model': name,
                'Model_Type': 'Neural Network' if 'MLP' in name else
                             'Ensemble' if any(x in name for x in ['Random Forest', 'Gradient', 'AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']) else
                             'Linear' if 'Logistic' in name else
                             'SVM' if 'SVM' in name else
                             'Tree' if 'Decision Tree' in name else
                             'Bayesian' if 'Naive Bayes' in name else 'Other',
                'CV_Mean': f"{cv_scores.mean():.4f}",
                'CV_Std': f"{cv_scores.std():.4f}",
                'Accuracy': f"{accuracy:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'Specificity': f"{specificity:.4f}",
                'Balanced_Acc': f"{balanced_acc:.4f}",
                'F1_Score': f"{f1:.4f}",
                'ROC_AUC': f"{roc_auc:.4f}",
                'PR_AUC': f"{pr_auc:.4f}",
                'MCC': f"{mcc:.4f}",
                'Kappa': f"{kappa:.4f}",
                'Training_Time': training_time
            })
            
            # Lưu kết quả cho vẽ biểu đồ
            self.models_results[name] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc,
                'cv_scores': cv_scores
            }
        
        results_df = pd.DataFrame(results)

        # Sắp xếp theo Balanced Accuracy (thay vì chỉ Accuracy)
        results_df = results_df.sort_values('Balanced_Acc', ascending=False)

        print("\n" + "="*180)
        print("BẢNG SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH PHÂN LOẠI ALZHEIMER")
        print("="*180)

        # Hiển thị kết quả theo nhóm metrics quan trọng
        display_cols = ['Model', 'Model_Type', 'CV_Mean', 'CV_Std', 'Accuracy', 'Balanced_Acc',
                       'Precision', 'Recall', 'Specificity', 'F1_Score', 'ROC_AUC', 'PR_AUC', 'MCC']

        print(results_df[display_cols].to_string(index=False))

        # Phân tích chi tiết
        print(f"\n📊 Tóm tắt hiệu suất mô hình:")
        print(f"   • Tổng số mô hình đánh giá: {len(results_df)}")
        print(f"   • Mô hình tốt nhất (Balanced Acc): {results_df.iloc[0]['Model']} ({results_df.iloc[0]['Balanced_Acc']})")
        print(f"   • Độ ổn định cao nhất (CV Std min): {results_df.loc[results_df['CV_Std'].astype(float).idxmin()]['Model']} (Std={results_df['CV_Std'].astype(float).min():.4f})")

        # Phân tích theo loại mô hình
        model_types = results_df.groupby('Model_Type')
        print("🔍 Phân tích theo loại mô hình:")
        for model_type, group in model_types:
            avg_balanced_acc = group['Balanced_Acc'].astype(float).mean()
            avg_roc_auc = group['ROC_AUC'].astype(float).mean()
            best_model = group.loc[group['Balanced_Acc'].astype(float).idxmax()]['Model']

            print(f"   • {model_type}: Trung bình Balanced Acc = {avg_balanced_acc:.4f}, "
                  f"ROC AUC = {avg_roc_auc:.4f}, Best: {best_model}")

        # Đánh giá clinical utility
        print("🏥 Clinical Utility Analysis:")
        for _, row in results_df.iterrows():
            balanced_acc = float(row['Balanced_Acc'])
            roc_auc = float(row['ROC_AUC'])

            if balanced_acc > 0.8 and roc_auc > 0.8:
                utility = "Excellent"
            elif balanced_acc > 0.75 and roc_auc > 0.75:
                utility = "Good"
            elif balanced_acc > 0.7 and roc_auc > 0.7:
                utility = "Fair"
            else:
                utility = "Poor"

            print(f"   • {row['Model']}: {utility} clinical utility (Balanced Acc: {balanced_acc:.3f}, ROC AUC: {roc_auc:.3f})")
        
        return results_df
    
    def plot_feature_comparison(self):
        """Vẽ biểu đồ so sánh đặc trưng với nhiều loại biểu đồ"""
        print("\n📈 Vẽ biểu đồ so sánh đặc trưng chi tiết...")

        # Chọn các đặc trưng quan trọng để so sánh
        key_features = {
            'linguistic': ['avg_words_per_sentence', 'sentence_length', 'lexical_diversity', 'semantic_fluency'],
            'acoustic': ['speech_rate', 'mean_pause_length', 'f0_mean', 'jitter', 'shimmer'],
            'cognitive': ['final_mmse_score', 'memory_score', 'cognitive_score']
        }

        # Tạo grid layout lớn hơn cho nhiều biểu đồ
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('PHÂN TÍCH ĐẶC TRƯNG NGÔN NGỮ VÀ ÂM HỌC: ALZHEIMER vs CONTROL', fontsize=16, y=0.95)

        # 1. Box plots cho từng nhóm đặc trưng
        plot_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

        subplot_idx = 0
        for category, features in key_features.items():
            available_features = [f for f in features if f in self.data.columns and
                                self.data[f].notna().sum() > 5]

            if available_features:
                ax = plt.subplot2grid((3, 3), plot_positions[subplot_idx])
                self._plot_feature_boxplots(ax, available_features, category.title())
                subplot_idx += 1

        # 2. Violin plots cho đặc trưng quan trọng
        if subplot_idx < 8:
            ax = plt.subplot2grid((3, 3), plot_positions[subplot_idx])
            self._plot_violin_features(ax)
            subplot_idx += 1

        # 3. Radar chart cho đặc trưng tổng hợp
        if subplot_idx < 8:
            ax = plt.subplot2grid((3, 3), (2, 2))
            self._plot_radar_features(ax)

        plt.tight_layout()
        plt.show()

        # Vẽ thêm biểu đồ correlation matrix
        self._plot_correlation_heatmap()

    def _plot_feature_boxplots(self, ax, features, category_name):
        """Vẽ box plots cho nhóm đặc trưng"""
        try:
            # Chuẩn bị dữ liệu
            plot_data = []
            plot_features = []

            for feature in features[:3]:  # Max 3 features per plot
                if feature in self.data.columns:
                    temp_data = self.data[['group', feature]].copy()
                    temp_data = temp_data.dropna()
                    if len(temp_data) > 0:
                        temp_data['Feature'] = feature.replace('_', ' ').title()
                        plot_data.append(temp_data)
                        plot_features.append(feature.replace('_', ' ').title())

            if plot_data:
                combined_data = pd.concat(plot_data, ignore_index=True)

                sns.boxplot(data=combined_data, x='Feature', y=feature, hue='group',
                           ax=ax, palette=['lightcoral', 'lightblue'])

                ax.set_title(f'{category_name} Features')
                ax.set_xlabel('Features')
                ax.set_ylabel('Values')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.text(0.5, 0.5, f'No data for {category_name}', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_violin_features(self, ax):
        """Vẽ violin plots cho đặc trưng quan trọng"""
        try:
            # Chọn 2-3 đặc trưng quan trọng nhất
            priority_features = ['lexical_diversity', 'speech_rate', 'mean_pause_length', 'final_mmse_score']
            available_features = [f for f in priority_features if f in self.data.columns]

            if not available_features:
                ax.text(0.5, 0.5, 'No features available for violin plot', ha='center', va='center', transform=ax.transAxes)
                return

            # Tạo dữ liệu cho violin plot
            plot_data = []
            for feature in available_features[:3]:
                temp_data = self.data[['group', feature]].copy()
                temp_data = temp_data.dropna()
                temp_data['Feature'] = feature.replace('_', ' ').title()
                plot_data.append(temp_data)

            if plot_data:
                combined_data = pd.concat(plot_data, ignore_index=True)

                sns.violinplot(data=combined_data, x='Feature', y=available_features[0],
                              hue='group', ax=ax, split=True, palette=['lightcoral', 'lightblue'])

                ax.set_title('Distribution Comparison (Violin Plot)')
                ax.set_xlabel('Features')
                ax.set_ylabel('Values')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.text(0.5, 0.5, 'No data for violin plot', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_radar_features(self, ax):
        """Vẽ radar chart cho đặc trưng tổng hợp"""
        try:
            # Tính giá trị trung bình cho từng nhóm
            alzheimer_means = {}
            control_means = {}

            radar_features = ['lexical_diversity', 'speech_rate', 'mean_pause_length',
                             'final_mmse_score', 'jitter', 'shimmer']

            for feature in radar_features:
                if feature in self.data.columns:
                    alz_data = self.data[self.data['group'] == 'Alzheimer'][feature].dropna()
                    ctrl_data = self.data[self.data['group'] == 'Control'][feature].dropna()

                    if len(alz_data) > 0:
                        alzheimer_means[feature] = alz_data.mean()
                    if len(ctrl_data) > 0:
                        control_means[feature] = ctrl_data.mean()

            if alzheimer_means and control_means:
                # Normalize values to 0-1 scale for radar chart
                features = list(set(alzheimer_means.keys()) & set(control_means.keys()))
                if len(features) >= 3:
                    # Simple normalization (can be improved)
                    alz_values = [alzheimer_means[f] for f in features]
                    ctrl_values = [control_means[f] for f in features]

                    # Create radar chart
                    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
                    angles += angles[:1]  # Close the loop

                    alz_values += alz_values[:1]
                    ctrl_values += ctrl_values[:1]

                    ax.plot(angles, alz_values, 'ro-', linewidth=2, label='Alzheimer', markersize=6)
                    ax.plot(angles, ctrl_values, 'bo-', linewidth=2, label='Control', markersize=6)
                    ax.fill(angles, alz_values, 'r', alpha=0.1)
                    ax.fill(angles, ctrl_values, 'b', alpha=0.1)

                    feature_labels = [f.replace('_', ' ').title() for f in features]
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(feature_labels, fontsize=8)
                    ax.set_title('Feature Profile Comparison (Radar)', fontsize=10)
                    ax.legend(loc='upper right', fontsize=8)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Need ≥3 features for radar', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for radar', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Radar error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_correlation_heatmap(self):
        """Vẽ heatmap correlation matrix"""
        try:
            print("\n🔗 Vẽ ma trận tương quan...")

            # Chọn các đặc trưng để phân tích tương quan
            corr_features = [
                'age', 'final_mmse_score', 'memory_score', 'cognitive_score',
                'avg_words_per_sentence', 'lexical_diversity', 'semantic_fluency',
                'speech_rate', 'mean_pause_length', 'f0_mean', 'jitter', 'shimmer'
            ]

            available_features = [f for f in corr_features if f in self.data.columns]
            corr_data = self.data[available_features].corr()

            if len(available_features) > 2:
                plt.figure(figsize=(12, 10))

                # Tạo mask cho upper triangle
                mask = np.triu(np.ones_like(corr_data, dtype=bool))

                sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                           fmt='.2f', annot_kws={"size": 8})

                plt.title('Ma Trận Tương Quan Các Đặc Trưng', fontsize=14, pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()

                # Phân tích tương quan mạnh
                print("\n🔍 Phân tích tương quan mạnh (|r| > 0.5):")
                strong_corr = []
                for i in range(len(corr_data.columns)):
                    for j in range(i+1, len(corr_data.columns)):
                        corr_val = corr_data.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            strong_corr.append({
                                'feature1': corr_data.columns[i],
                                'feature2': corr_data.columns[j],
                                'correlation': corr_val
                            })

                if strong_corr:
                    for corr in sorted(strong_corr, key=lambda x: abs(x['correlation']), reverse=True):
                        print(f"   • {corr['feature1']} ↔ {corr['feature2']}: r = {corr['correlation']:.3f}")
                else:
                    print("   • Không có tương quan mạnh đáng kể")

            else:
                print("⚠️ Không đủ features để vẽ correlation matrix")

        except Exception as e:
            print(f"❌ Lỗi vẽ correlation heatmap: {e}")
    
    def plot_roc_curves(self):
        """Vẽ ROC curves cho các mô hình"""
        print("\n📊 Vẽ ROC curves...")
        
        if not self.models_results:
            print("⚠️ Chưa có kết quả mô hình để vẽ ROC curves")
            return
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (name, results) in enumerate(self.models_results.items()):
            if 'fpr' in results and 'tpr' in results:
                plt.plot(results['fpr'], results['tpr'], 
                        color=colors[i % len(colors)], lw=2, 
                        label=f'{name} (AUC = {results["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - So sánh Mô hình Phân loại Alzheimer')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_confusion_matrices(self):
        """Vẽ confusion matrices"""
        print("\n🔄 Vẽ Confusion Matrices...")
        
        if not self.models_results:
            print("⚠️ Chưa có kết quả mô hình để vẽ confusion matrices")
            return
        
        n_models = len(self.models_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Confusion Matrices - Các Mô hình', fontsize=16)
        
        for i, (name, results) in enumerate(self.models_results.items()):
            if 'y_test' in results and 'y_pred' in results:
                cm = confusion_matrix(results['y_test'], results['y_pred'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{name}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
                axes[i].set_xticklabels(['Control', 'Alzheimer'])
                axes[i].set_yticklabels(['Control', 'Alzheimer'])
        
        plt.tight_layout()
        plt.show()
    
    def plot_violin_comparison(self):
        """Vẽ violin plot cho các đặc trưng quan trọng"""
        print("\n🎻 Vẽ violin plot...")
        
        key_features = ['speech_rate', 'lexical_diversity', 'mean_pause_length', 'jitter']
        available_features = [f for f in key_features if f in self.data.columns]
        
        if not available_features:
            print("⚠️ Không có features phù hợp cho violin plot")
            return
        
        n_features = len(available_features)
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
        if n_features == 1:
            axes = [axes]
        
        fig.suptitle('Phân phối Đặc trưng - Violin Plot', fontsize=16)
        
        for i, feature in enumerate(available_features):
            try:
                sns.violinplot(data=self.data, x='group', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature.replace("_", " ").title()}')
                axes[i].set_xlabel('Nhóm')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Lỗi: {str(e)}', ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_analysis(self):
        """Vẽ biểu đồ tương quan"""
        print("\n🔗 Phân tích tương quan...")
        
        # Chọn các đặc trưng số
        numeric_features = []
        potential_features = [
            'age', 'mmse_score', 'speech_rate', 'lexical_diversity',
            'mean_pause_length', 'jitter', 'shimmer', 'semantic_fluency',
            'f0_mean', 'hnr'
        ]
        
        for feature in potential_features:
            if feature in self.data.columns and self.data[feature].notna().sum() > 10:
                numeric_features.append(feature)
        
        if len(numeric_features) < 2:
            print("⚠️ Không đủ features số để phân tích tương quan")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plots nếu có đủ dữ liệu
        if 'mmse_score' in numeric_features and 'lexical_diversity' in numeric_features:
            colors = ['red' if x == 'Alzheimer' else 'blue' for x in self.data['group']]
            axes[0].scatter(self.data['mmse_score'], self.data['lexical_diversity'], 
                           c=colors, alpha=0.6)
            axes[0].set_xlabel('MMSE Score')
            axes[0].set_ylabel('Lexical Diversity')
            axes[0].set_title('Tương quan MMSE vs Lexical Diversity')
            
            # Tạo legend
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Alzheimer')
            blue_patch = mpatches.Patch(color='blue', label='Control')
            axes[0].legend(handles=[red_patch, blue_patch])
        else:
            axes[0].text(0.5, 0.5, 'Không có dữ liệu MMSE/Lexical Diversity', 
                        ha='center', va='center', transform=axes[0].transAxes)
        
        if 'age' in numeric_features and 'speech_rate' in numeric_features:
            colors = ['red' if x == 'Alzheimer' else 'blue' for x in self.data['group']]
            axes[1].scatter(self.data['age'], self.data['speech_rate'], 
                           c=colors, alpha=0.6)
            axes[1].set_xlabel('Tuổi')
            axes[1].set_ylabel('Tốc độ Nói (words/min)')
            axes[1].set_title('Tương quan Tuổi vs Tốc độ Nói')
        else:
            axes[1].text(0.5, 0.5, 'Không có dữ liệu Age/Speech Rate', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation matrix
        if len(numeric_features) > 2:
            plt.figure(figsize=(12, 10))
            correlation_data = self.data[numeric_features].corr()
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Ma trận Tương quan các Đặc trưng')
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Không đủ features để vẽ correlation matrix")
    
    def plot_longitudinal_trends(self):
        """Vẽ biểu đồ xu hướng theo thời gian với dữ liệu thực tế"""
        print("\n📈 Vẽ xu hướng tiến triển theo thời gian...")

        # Thử load dữ liệu longitudinal từ database/API trước
        longitudinal_data = self._load_longitudinal_data()

        if longitudinal_data is not None and not longitudinal_data.empty:
            print("📅 Sử dụng dữ liệu longitudinal thực tế")
            self._plot_real_longitudinal_data(longitudinal_data)
        else:
            print("📅 Không có dữ liệu longitudinal thực tế, tạo dữ liệu giả lập")
            self._plot_simulated_longitudinal_data()

    def _load_longitudinal_data(self):
        """Load dữ liệu longitudinal từ database"""
        try:
            # Kiểm tra xem có thể kết nối database không
            if hasattr(self, 'data') and not self.data.empty:
                # Tạo dữ liệu longitudinal giả lập từ dữ liệu hiện có
                # Trong thực tế, bạn sẽ query database để lấy dữ liệu theo thời gian

                # Giả sử chúng ta có thể tạo dữ liệu longitudinal từ session_id và created_at
                if 'created_at' in self.data.columns and 'session_id' in self.data.columns:
                    # Nhóm dữ liệu theo user và sắp xếp theo thời gian
                    long_data = self.data.copy()
                    long_data['assessment_date'] = pd.to_datetime(long_data['created_at'])
                    long_data = long_data.sort_values(['user_id', 'assessment_date'])

                    # Tính số ngày từ assessment đầu tiên
                    long_data['days_from_first'] = long_data.groupby('user_id')['assessment_date'].transform(
                        lambda x: (x - x.min()).dt.days
                    )

                    # Chỉ giữ lại users có ít nhất 2 assessments
                    user_counts = long_data['user_id'].value_counts()
                    valid_users = user_counts[user_counts >= 2].index
                    long_data = long_data[long_data['user_id'].isin(valid_users)]

                    if len(long_data) > 0:
                        print(f"✅ Tìm thấy {len(valid_users)} users với dữ liệu longitudinal")
                        return long_data

            return None

        except Exception as e:
            print(f"⚠️ Lỗi load dữ liệu longitudinal: {e}")
            return None

    def _plot_real_longitudinal_data(self, long_data):
        """Vẽ biểu đồ với dữ liệu longitudinal thực tế"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Xu Hướng Tiến Triển Theo Thời Gian (Dữ Liệu Thực Tế)', fontsize=16)

            # 1. MMSE Score progression
            if 'final_mmse_score' in long_data.columns:
                self._plot_longitudinal_feature(axes[0, 0], long_data, 'final_mmse_score',
                                              'MMSE Score', 'MMSE Score')

            # 2. Lexical Diversity progression
            if 'lexical_diversity' in long_data.columns:
                self._plot_longitudinal_feature(axes[0, 1], long_data, 'lexical_diversity',
                                              'Lexical Diversity', 'Lexical Diversity')

            # 3. Speech Rate progression
            if 'speech_rate' in long_data.columns:
                self._plot_longitudinal_feature(axes[1, 0], long_data, 'speech_rate',
                                              'Speech Rate (words/min)', 'Tốc độ Nói')

            # 4. Mean Pause Length progression
            if 'mean_pause_length' in long_data.columns:
                self._plot_longitudinal_feature(axes[1, 1], long_data, 'mean_pause_length',
                                              'Mean Pause Length (s)', 'Độ Dài Dừng Trung Bình')

            plt.tight_layout()
            plt.show()

            # Vẽ biểu đồ individual trajectories
            self._plot_individual_trajectories(long_data)

        except Exception as e:
            print(f"❌ Lỗi vẽ dữ liệu longitudinal thực tế: {e}")
            self._plot_simulated_longitudinal_data()

    def _plot_longitudinal_feature(self, ax, data, feature_col, ylabel, title):
        """Vẽ biểu đồ xu hướng cho một đặc trưng"""
        try:
            # Lấy dữ liệu theo nhóm
            alz_data = data[data['group'] == 'Alzheimer']
            ctrl_data = data[data['group'] == 'Control']

            # Tính trung bình theo thời gian (theo tháng)
            data_copy = data.copy()
            data_copy['months_from_first'] = data_copy['days_from_first'] / 30.44  # Convert to months

            # Group by months and calculate means
            alz_monthly = alz_data.groupby(pd.cut(alz_data['months_from_first'],
                                                 bins=range(0, 25, 3), right=False))[feature_col].agg(['mean', 'std', 'count']).dropna()
            ctrl_monthly = ctrl_data.groupby(pd.cut(ctrl_data['months_from_first'],
                                                  bins=range(0, 25, 3), right=False))[feature_col].agg(['mean', 'std', 'count']).dropna()

            if not alz_monthly.empty and not ctrl_monthly.empty:
                # Plot with error bars
                months = range(0, 24, 3)

                # Alzheimer group
                alz_means = []
                alz_stds = []
                for i, month in enumerate(months[:-1]):
                    if i < len(alz_monthly):
                        alz_means.append(alz_monthly.iloc[i]['mean'])
                        alz_stds.append(alz_monthly.iloc[i]['std'])
                    else:
                        alz_means.append(np.nan)
                        alz_stds.append(np.nan)

                # Control group
                ctrl_means = []
                ctrl_stds = []
                for i, month in enumerate(months[:-1]):
                    if i < len(ctrl_monthly):
                        ctrl_means.append(ctrl_monthly.iloc[i]['mean'])
                        ctrl_stds.append(ctrl_monthly.iloc[i]['std'])
                    else:
                        ctrl_means.append(np.nan)
                        ctrl_stds.append(np.nan)

                # Plot lines
                ax.errorbar(months[:-1], alz_means, yerr=alz_stds, fmt='ro-',
                           label='Alzheimer', linewidth=2, markersize=6, capsize=3)
                ax.errorbar(months[:-1], ctrl_means, yerr=ctrl_stds, fmt='bo-',
                           label='Control', linewidth=2, markersize=6, capsize=3)

                ax.set_xlabel('Thời gian (tháng)')
                ax.set_ylabel(ylabel)
                ax.set_title(f'Tiến triển {title}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Không đủ dữ liệu\n{title}', ha='center', va='center', transform=ax.transAxes)

        except Exception as e:
            ax.text(0.5, 0.5, f'Lỗi: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_individual_trajectories(self, long_data):
        """Vẽ biểu đồ quỹ đạo cá nhân"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Quỹ Đạo Cá Nhân Theo Thời Gian', fontsize=16)

            features_to_plot = ['final_mmse_score', 'lexical_diversity', 'speech_rate']
            feature_labels = ['MMSE Score', 'Lexical Diversity', 'Speech Rate']

            for i, (feature, label) in enumerate(zip(features_to_plot, feature_labels)):
                if feature in long_data.columns:
                    # Lấy top 5 users từ mỗi nhóm để vẽ
                    alz_users = long_data[long_data['group'] == 'Alzheimer']['user_id'].unique()[:5]
                    ctrl_users = long_data[long_data['group'] == 'Control']['user_id'].unique()[:5]

                    # Vẽ quỹ đạo cho Alzheimer group
                    for user in alz_users:
                        user_data = long_data[(long_data['user_id'] == user) &
                                            (long_data['group'] == 'Alzheimer')].sort_values('days_from_first')
                        if len(user_data) >= 2:
                            axes[i].plot(user_data['days_from_first'] / 30.44,  # Convert to months
                                       user_data[feature], 'r-', alpha=0.7, linewidth=1)

                    # Vẽ quỹ đạo cho Control group
                    for user in ctrl_users:
                        user_data = long_data[(long_data['user_id'] == user) &
                                            (long_data['group'] == 'Control')].sort_values('days_from_first')
                        if len(user_data) >= 2:
                            axes[i].plot(user_data['days_from_first'] / 30.44,  # Convert to months
                                       user_data[feature], 'b-', alpha=0.7, linewidth=1)

                    axes[i].set_xlabel('Thời gian (tháng)')
                    axes[i].set_ylabel(label)
                    axes[i].set_title(f'Quỹ đạo {label}')
                    axes[i].grid(True, alpha=0.3)

            # Add legend
            from matplotlib.patches import Patch
            red_patch = Patch(color='red', label='Alzheimer', alpha=0.7)
            blue_patch = Patch(color='blue', label='Control', alpha=0.7)
            axes[0].legend(handles=[red_patch, blue_patch], loc='best')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️ Lỗi vẽ quỹ đạo cá nhân: {e}")

    def _plot_simulated_longitudinal_data(self):
        """Vẽ biểu đồ với dữ liệu giả lập khi không có dữ liệu thực tế"""
        print("📊 Tạo dữ liệu giả lập cho phân tích longitudinal...")

        # Lấy giá trị trung bình từ dữ liệu thực
        alz_data = self.data[self.data['group'] == 'Alzheimer']
        ctrl_data = self.data[self.data['group'] == 'Control']

        time_points = [0, 6, 12, 18, 24]  # tháng

        # MMSE progression
        if 'final_mmse_score' in self.data.columns and len(alz_data) > 0 and len(ctrl_data) > 0:
            alz_mmse_base = alz_data['final_mmse_score'].mean()
            ctrl_mmse_base = ctrl_data['final_mmse_score'].mean()

            # Giả lập sự suy giảm dựa trên nghiên cứu thực tế
            alzheimer_mmse = [alz_mmse_base, alz_mmse_base-2, alz_mmse_base-5,
                            alz_mmse_base-8, alz_mmse_base-12]
            control_mmse = [ctrl_mmse_base, ctrl_mmse_base-0.5, ctrl_mmse_base-1,
                          ctrl_mmse_base-1.5, ctrl_mmse_base-2]
        else:
            alzheimer_mmse = [22, 20, 17, 14, 11]
            control_mmse = [28, 28, 27, 27, 26]

        # Lexical diversity progression
        if 'lexical_diversity' in self.data.columns and len(alz_data) > 0 and len(ctrl_data) > 0:
            alz_lex_base = alz_data['lexical_diversity'].mean()
            ctrl_lex_base = ctrl_data['lexical_diversity'].mean()

            alzheimer_lexical = [alz_lex_base, alz_lex_base-0.05, alz_lex_base-0.10,
                               alz_lex_base-0.17, alz_lex_base-0.23]
            control_lexical = [ctrl_lex_base, ctrl_lex_base-0.01, ctrl_lex_base-0.02,
                             ctrl_lex_base-0.03, ctrl_lex_base-0.04]
        else:
            alzheimer_lexical = [0.55, 0.50, 0.45, 0.38, 0.32]
            control_lexical = [0.65, 0.64, 0.63, 0.62, 0.61]

        # Speech rate progression
        if 'speech_rate' in self.data.columns and len(alz_data) > 0 and len(ctrl_data) > 0:
            alz_speech_base = alz_data['speech_rate'].mean()
            ctrl_speech_base = ctrl_data['speech_rate'].mean()

            alzheimer_speech_rate = [alz_speech_base, alz_speech_base-7, alz_speech_base-15,
                                   alz_speech_base-27, alz_speech_base-40]
            control_speech_rate = [ctrl_speech_base, ctrl_speech_base-2, ctrl_speech_base-4,
                                 ctrl_speech_base-5, ctrl_speech_base-7]
        else:
            alzheimer_speech_rate = [125, 118, 110, 98, 85]
            control_speech_rate = [140, 138, 136, 135, 133]

        # Mean pause length progression
        if 'mean_pause_length' in self.data.columns and len(alz_data) > 0 and len(ctrl_data) > 0:
            alz_pause_base = alz_data['mean_pause_length'].mean()
            ctrl_pause_base = ctrl_data['mean_pause_length'].mean()

            alzheimer_pause = [alz_pause_base, alz_pause_base+0.1, alz_pause_base+0.25,
                             alz_pause_base+0.45, alz_pause_base+0.7]
            control_pause = [ctrl_pause_base, ctrl_pause_base+0.02, ctrl_pause_base+0.05,
                           ctrl_pause_base+0.08, ctrl_pause_base+0.1]
        else:
            alzheimer_pause = [0.75, 0.85, 1.0, 1.2, 1.45]
            control_pause = [0.48, 0.5, 0.53, 0.56, 0.58]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Xu Hướng Tiến Triển Theo Thời Gian (Dữ Liệu Giả Lập)', fontsize=16)

        # MMSE Score
        axes[0, 0].plot(time_points, alzheimer_mmse, 'ro-', label='Alzheimer', linewidth=2, markersize=6)
        axes[0, 0].plot(time_points, control_mmse, 'bo-', label='Control', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Thời gian (tháng)')
        axes[0, 0].set_ylabel('MMSE Score')
        axes[0, 0].set_title('Tiến triển MMSE Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 30)

        # Lexical Diversity
        axes[0, 1].plot(time_points, alzheimer_lexical, 'ro-', label='Alzheimer', linewidth=2, markersize=6)
        axes[0, 1].plot(time_points, control_lexical, 'bo-', label='Control', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Thời gian (tháng)')
        axes[0, 1].set_ylabel('Lexical Diversity')
        axes[0, 1].set_title('Tiến triển Lexical Diversity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        # Speech Rate
        axes[1, 0].plot(time_points, alzheimer_speech_rate, 'ro-', label='Alzheimer', linewidth=2, markersize=6)
        axes[1, 0].plot(time_points, control_speech_rate, 'bo-', label='Control', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Thời gian (tháng)')
        axes[1, 0].set_ylabel('Speech Rate (words/min)')
        axes[1, 0].set_title('Tiến triển Tốc độ Nói')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Mean Pause Length
        axes[1, 1].plot(time_points, alzheimer_pause, 'ro-', label='Alzheimer', linewidth=2, markersize=6)
        axes[1, 1].plot(time_points, control_pause, 'bo-', label='Control', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Thời gian (tháng)')
        axes[1, 1].set_ylabel('Mean Pause Length (s)')
        axes[1, 1].set_title('Tiến triển Độ Dài Dừng')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """Vẽ biểu đồ tầm quan trọng của features"""
        print("\n📊 Phân tích tầm quan trọng features...")
        
        if 'Random Forest' not in self.models_results:
            print("⚠️ Cần mô hình Random Forest để phân tích feature importance")
            return
        
        # Lấy mô hình Random Forest đã train
        try:
            # Chuẩn bị dữ liệu lại
            feature_columns = [col for col in self.data.columns 
                              if col not in ['group', 'label', 'user_id', 'gender', 'diagnosis', 'diagnosis_clean']]
            
            numeric_features = []
            for col in feature_columns:
                if self.data[col].dtype in ['int64', 'float64']:
                    numeric_features.append(col)
            
            X = self.data[numeric_features].fillna(self.data[numeric_features].median())
            y = self.data['label']
            
            # Train lại Random Forest để lấy feature importance
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_model.fit(X, y)
            
            # Lấy feature importance
            feature_importance = pd.DataFrame({
                'feature': numeric_features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            # Vẽ biểu đồ
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(feature_importance)), feature_importance['importance'])
            plt.yticks(range(len(feature_importance)), 
                      [f.replace('_', ' ').title() for f in feature_importance['feature']])
            plt.xlabel('Feature Importance')
            plt.title('Tầm quan trọng của các Đặc trưng (Random Forest)')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Lỗi phân tích feature importance: {e}")
    
    def export_results_to_csv(self):
        """Xuất kết quả ra file CSV"""
        print("\n💾 Xuất kết quả ra file...")

        try:
            # Tạo thư mục output nếu chưa có
            import os
            output_dir = 'analysis_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Xuất bảng so sánh features
            feature_table = self.create_feature_comparison_table()
            feature_table.to_csv(f'{output_dir}/feature_comparison_results.csv', index=False, encoding='utf-8-sig')
            print("✅ Đã xuất feature_comparison_results.csv")

            # Xuất kết quả models
            if self.models_results:
                model_results = self.train_models_properly()
                model_results.to_csv(f'{output_dir}/model_performance_results.csv', index=False, encoding='utf-8-sig')
                print("✅ Đã xuất model_performance_results.csv")

            # Xuất dữ liệu đã xử lý
            self.data.to_csv(f'{output_dir}/processed_alzheimer_data.csv', index=False, encoding='utf-8-sig')
            print("✅ Đã xuất processed_alzheimer_data.csv")

        except Exception as e:
            print(f"❌ Lỗi xuất file: {e}")

    def export_comprehensive_report(self, output_dir='analysis_results'):
        """Xuất báo cáo tổng hợp với nhiều định dạng"""
        print(f"\n📊 Xuất báo cáo tổng hợp ra thư mục: {output_dir}")

        try:
            import os
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 1. Xuất dữ liệu CSV
            self._export_data_csv(output_dir)

            # 2. Xuất kết quả phân tích thống kê
            self._export_statistical_analysis(output_dir)

            # 3. Xuất biểu đồ
            self._export_charts(output_dir)

            # 4. Tạo báo cáo PDF/HTML (nếu có thể)
            self._export_pdf_report(output_dir)

            print(f"✅ Đã xuất báo cáo tổng hợp vào thư mục '{output_dir}'")

        except Exception as e:
            print(f"❌ Lỗi xuất báo cáo tổng hợp: {e}")

    def _export_data_csv(self, output_dir):
        """Xuất dữ liệu ra các file CSV"""
        try:
            # Dữ liệu gốc đã xử lý
            self.data.to_csv(f'{output_dir}/processed_data.csv', index=False, encoding='utf-8-sig')

            # Dữ liệu theo nhóm
            alzheimer_data = self.data[self.data['group'] == 'Alzheimer']
            control_data = self.data[self.data['group'] == 'Control']

            alzheimer_data.to_csv(f'{output_dir}/alzheimer_group_data.csv', index=False, encoding='utf-8-sig')
            control_data.to_csv(f'{output_dir}/control_group_data.csv', index=False, encoding='utf-8-sig')

            # Thống kê tóm tắt
            summary_stats = self.data.groupby('group').agg({
                'final_mmse_score': ['count', 'mean', 'std', 'min', 'max'],
                'age': ['mean', 'std'],
                'lexical_diversity': ['mean', 'std'],
                'speech_rate': ['mean', 'std'],
                'mean_pause_length': ['mean', 'std']
            }).round(3)

            summary_stats.to_csv(f'{output_dir}/group_summary_statistics.csv', encoding='utf-8-sig')

            print("✅ Đã xuất dữ liệu CSV")

        except Exception as e:
            print(f"❌ Lỗi xuất dữ liệu CSV: {e}")

    def _export_statistical_analysis(self, output_dir):
        """Xuất kết quả phân tích thống kê"""
        try:
            # Bảng so sánh đặc trưng
            feature_comparison = self.create_feature_comparison_table()
            feature_comparison.to_csv(f'{output_dir}/statistical_comparison.csv', index=False, encoding='utf-8-sig')

            # Kết quả mô hình
            if self.models_results:
                model_performance = self.train_models_properly()
                model_performance.to_csv(f'{output_dir}/model_performance_comparison.csv', index=False, encoding='utf-8-sig')

            # Ma trận tương quan
            if len(self.data.select_dtypes(include=[np.number]).columns) > 2:
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                correlation_matrix = self.data[numeric_cols].corr()
                correlation_matrix.to_csv(f'{output_dir}/correlation_matrix.csv', encoding='utf-8-sig')

            print("✅ Đã xuất kết quả phân tích thống kê")

        except Exception as e:
            print(f"❌ Lỗi xuất phân tích thống kê: {e}")

    def _export_charts(self, output_dir):
        """Xuất biểu đồ ra file PNG"""
        try:
            # Vẽ và lưu các biểu đồ
            plt.style.use('default')  # Reset style

            # 1. Box plots
            self._save_box_plots(output_dir)

            # 2. ROC curves
            self._save_roc_curves(output_dir)

            # 3. Confusion matrices
            self._save_confusion_matrices(output_dir)

            # 4. Correlation heatmap
            self._save_correlation_heatmap(output_dir)

            # 5. Feature importance (nếu có)
            self._save_feature_importance(output_dir)

            print("✅ Đã xuất biểu đồ")

        except Exception as e:
            print(f"❌ Lỗi xuất biểu đồ: {e}")

    def _save_box_plots(self, output_dir):
        """Lưu box plots"""
        try:
            key_features = ['final_mmse_score', 'lexical_diversity', 'speech_rate', 'mean_pause_length']

            n_features = len([f for f in key_features if f in self.data.columns])
            if n_features > 0:
                fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))

                if n_features == 1:
                    axes = [axes]

                for i, feature in enumerate(key_features):
                    if feature in self.data.columns and i < n_features:
                        sns.boxplot(data=self.data, x='group', y=feature, ax=axes[i],
                                  palette=['lightcoral', 'lightblue'])
                        axes[i].set_title(f'{feature.replace("_", " ").title()}')

                plt.tight_layout()
                plt.savefig(f'{output_dir}/feature_boxplots.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"⚠️ Lỗi lưu box plots: {e}")

    def _save_roc_curves(self, output_dir):
        """Lưu ROC curves"""
        try:
            if not self.models_results:
                return

            plt.figure(figsize=(10, 8))
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

            for i, (name, results) in enumerate(self.models_results.items()):
                if 'fpr' in results and 'tpr' in results and results['fpr'] is not None:
                    plt.plot(results['fpr'], results['tpr'],
                            color=colors[i % len(colors)], lw=2,
                            label=f'{name} (AUC = {results["auc"]:.3f})')

            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - Model Comparison')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️ Lỗi lưu ROC curves: {e}")

    def _save_confusion_matrices(self, output_dir):
        """Lưu confusion matrices"""
        try:
            if not self.models_results:
                return

            n_models = len(self.models_results)
            if n_models > 6:  # Limit to 6 models for readability
                top_models = sorted(self.models_results.items(),
                                  key=lambda x: x[1].get('auc', 0), reverse=True)[:6]
                models_to_plot = dict(top_models)
            else:
                models_to_plot = self.models_results

            n_cols = min(3, len(models_to_plot))
            n_rows = (len(models_to_plot) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()

            for i, (name, results) in enumerate(models_to_plot.items()):
                if i < len(axes) and 'y_test' in results and 'y_pred' in results:
                    cm = confusion_matrix(results['y_test'], results['y_pred'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                    axes[i].set_title(f'{name}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                    axes[i].set_xticklabels(['Control', 'Alzheimer'])
                    axes[i].set_yticklabels(['Control', 'Alzheimer'])

            # Hide empty subplots
            for i in range(len(models_to_plot), len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️ Lỗi lưu confusion matrices: {e}")

    def _save_correlation_heatmap(self, output_dir):
        """Lưu correlation heatmap"""
        try:
            numeric_features = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 2:
                corr_data = self.data[numeric_features].corr()

                plt.figure(figsize=(12, 10))
                mask = np.triu(np.ones_like(corr_data, dtype=bool))
                sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, fmt='.2f')
                plt.title('Feature Correlation Matrix')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"⚠️ Lỗi lưu correlation heatmap: {e}")

    def _save_feature_importance(self, output_dir):
        """Lưu biểu đồ feature importance"""
        try:
            if 'Random Forest' not in self.models_results:
                return

            # Recreate Random Forest model for feature importance
            feature_columns = [col for col in self.data.columns
                              if col not in ['group', 'label', 'user_id', 'gender', 'diagnosis', 'diagnosis_clean']]
            numeric_features = [col for col in feature_columns if self.data[col].dtype in ['int64', 'float64']]
            X = self.data[numeric_features].fillna(self.data[numeric_features].median())
            y = self.data['label']

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X, y)

            # Plot feature importance
            feature_importance = pd.DataFrame({
                'feature': numeric_features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(feature_importance)), feature_importance['importance'])
            plt.yticks(range(len(feature_importance)),
                      [f.replace('_', ' ').title() for f in feature_importance['feature']])
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance')
            plt.tight_layout()

            plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️ Lỗi lưu feature importance: {e}")

    def _export_pdf_report(self, output_dir):
        """Tạo báo cáo PDF (nếu có thể)"""
        try:
            # Tạo file text report trước
            report_content = self._generate_text_report()

            with open(f'{output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
                f.write(report_content)

            print("✅ Đã tạo báo cáo text (PDF cần thư viện bổ sung)")

        except Exception as e:
            print(f"⚠️ Lỗi tạo báo cáo PDF: {e}")

    def _generate_text_report(self):
        """Tạo nội dung báo cáo text"""
        try:
            report = []
            report.append("="*80)
            report.append("BÁO CÁO PHÂN TÍCH HỆ THỐNG ĐÁNH GIÁ NHẬN THỨC ALZHEIMER")
            report.append("="*80)
            report.append(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")

            # Thông tin tổng quan
            report.append("1. THÔNG TIN TỔNG QUAN")
            report.append("-" * 30)
            report.append(f"Tổng số mẫu: {len(self.data)}")
            report.append(f"Nhóm Alzheimer: {len(self.data[self.data['group'] == 'Alzheimer'])}")
            report.append(f"Nhóm Control: {len(self.data[self.data['group'] == 'Control'])}")
            report.append("")

            # Kết quả phân tích đặc trưng
            report.append("2. PHÂN TÍCH ĐẶC TRƯNG")
            report.append("-" * 30)
            feature_table = self.create_feature_comparison_table()
            if not feature_table.empty:
                report.append("Top 5 đặc trưng quan trọng nhất:")
                top_features = feature_table.head(5)
                for _, row in top_features.iterrows():
                    report.append(f"• {row['Feature']}: Cohen's d = {row['Cohens_D']}, p = {row['P_Value']}")
            report.append("")

            # Kết quả mô hình
            report.append("3. HIỆU SUẤT MÔ HÌNH")
            report.append("-" * 30)
            if self.models_results:
                model_results = self.train_models_properly()
                if not model_results.empty:
                    best_model = model_results.iloc[0]
                    report.append(f"Mô hình tốt nhất: {best_model['Model']}")
                    report.append(f"Balanced Accuracy: {best_model['Balanced_Acc']}")
                    report.append(f"ROC AUC: {best_model['ROC_AUC']}")
            report.append("")

            # Kết luận
            report.append("4. KẾT LUẬN")
            report.append("-" * 30)
            report.append("Báo cáo này cung cấp cái nhìn tổng quan về khả năng phân biệt")
            report.append("giữa nhóm Alzheimer và nhóm đối chứng dựa trên các đặc trưng")
            report.append("ngôn ngữ và âm học.")
            report.append("")
            report.append("Các file kết quả đã được lưu trong thư mục analysis_results/")

            return "\n".join(report)

        except Exception as e:
            return f"Lỗi tạo báo cáo: {e}"
    
    def generate_comprehensive_report(self):
        """Tạo báo cáo tổng hợp"""
        print("\n📋 Tạo báo cáo tổng hợp...")
        print(f"📊 Tổng số mẫu: {len(self.data)}")
        print(f"📊 Phân bố nhãn: {self.data['group'].value_counts().to_dict()}")
        
        # 1. Bảng so sánh đặc trưng
        print("\n" + "="*50)
        print("1. BẢNG SO SÁNH ĐẶC TRƯNG")
        print("="*50)
        feature_table = self.create_feature_comparison_table()
        
        # 2. Bảng kết quả mô hình
        print("\n" + "="*50)
        print("2. KẾT QUẢ CÁC MÔ HÌNH")
        print("="*50)
        model_results = self.train_models_properly()
        
        # 3. Vẽ tất cả các biểu đồ
        print("\n" + "="*50)
        print("3. CÁC BIỂU ĐỒ PHÂN TÍCH")
        print("="*50)
        
        self.plot_feature_comparison()
        self.plot_violin_comparison()
        self.plot_roc_curves()
        self.plot_confusion_matrices()
        self.plot_correlation_analysis()
        self.plot_longitudinal_trends()
        self.plot_feature_importance()
        
        # 4. Xuất kết quả
        self.export_results_to_csv()
        
        print("\n" + "="*50)
        print("📋 TỔNG KẾT")
        print("="*50)
        print("✅ Hoàn thành phân tích toàn diện!")
        print("📊 Bảng so sánh đặc trưng đã được tạo với statistical test")
        print("🤖 Kết quả mô hình đã được đánh giá với cross-validation") 
        print("📈 Tất cả biểu đồ phân tích đã được vẽ")
        print("💾 Kết quả đã được xuất ra file CSV")
        
        return feature_table, model_results

# Hướng dẫn sử dụng với hệ thống thực
def main_with_real_system():
    """
    Hàm chính để chạy phân tích với dữ liệu thực
    """
    print("🚀 Bắt đầu phân tích Alzheimer với dữ liệu thực...")
    
    # Cấu hình kết nối
    config = {
        'db_path': 'cognitive_assessment.db',  # Đường dẫn database
        'api_url': 'http://localhost:5000/api',  # URL API backend
        'csv_backup': 'data/alzheimer_assessment_data.csv'  # File CSV dự phòng
    }
    
    # Khởi tạo analyzer với cấu hình thực
    analyzer = AlzheimerAnalysisVisualizer(
        db_path=config['db_path'],
        api_url=config['api_url'],
        config=config
    )
    
    # Tạo báo cáo tổng hợp
    feature_table, model_results = analyzer.generate_comprehensive_report()
    
    return analyzer, feature_table, model_results

def main():
    """Hàm chính để demo"""
    return main_with_real_system()

# Chạy phân tích
if __name__ == "__main__":
    analyzer, feature_table, model_results = main()