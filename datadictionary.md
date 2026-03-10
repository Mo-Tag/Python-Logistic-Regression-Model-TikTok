# Data Dictionary 📊
This data dictionary describes the columns in the tiktok_dataset.csv file used in the TikTok Verified Status Prediction project. The dataset contains information about TikTok videos, including metadata, engagement metrics, and user status. It is used to build a logistic regression model to predict whether a video's author is verified.

## Dataset Overview

- Rows: 19,382 (original, before preprocessing)
- Columns: 12 (original) + 1 derived feature (text_length)
- Target Variable: verified_status (binary: 'verified' or 'not verified')
- Key Preprocessing Steps:
      - Missing values: Dropped rows with NaNs (298 rows affected).
      - Outliers: Capped in video_like_count and video_comment_count using IQR method.
      - Class Imbalance: Upsampled minority class ('verified') to match majority.
      - Feature Engineering: Added text_length from video_transcription_text.
      - Encoding: One-hot encoded categorical features like claim_status and author_ban_status.


## Column Descriptions

| Column Name | Data Type | Description | Possible Values / Range | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `#` | `int64` | Row index / unique identifier. | `1` to `19,382` | Sequential ID; not used as a model feature. |
| `claim_status` | `object` | Content categorization. | `claim`, `opinion`, `NaN` | One-hot encoded; 'claims' drive higher engagement. |
| `video_id` | `int64` | Unique video identifier. | `~1.23e+09` to `~9.99e+09` | Dropped from models (no predictive value). |
| `video_duration_sec` | `int64` | Duration of the video. | `5` to `60` | Positive association with verification odds. |
| `video_transcription_text` | `object` | Transcribed audio content. | Free-form text | Used to derive `text_length`. |
| **`verified_status`** | `object` | **Target Variable**: Author status. | `verified`, `not verified` | Imbalanced (6.3% verified); binary encoded (1/0). |
| `author_ban_status` | `object` | Current status of the creator. | `active`, `under review`, `banned` | One-hot encoded for modeling. |
| `video_view_count` | `float64` | Total views received. | `20` to `999,817` | High correlation with likes (0.86). |
| `video_like_count` | `float64` | Total likes received. | `0` to `657,830` | Dropped in final models to avoid multicollinearity. |
| `video_share_count` | `float64` | Total shares. | `0` to `256,130` | Included in model; small positive coefficient. |
| `video_download_count` | `float64` | Total downloads. | `0` to `14,994` | Included in model; small negative coefficient. |
| `video_comment_count` | `float64` | Total comments. | `0` to `9,599` | Included in model; small negative coefficient. |
| `text_length` | `int64` | Character count of transcription. | Varies (Avg ~84-89) | Feature-engineered; optional for final models. |

## Additional Notes
- Correlations: Strong correlations exist between engagement metrics (e.g., views and likes: 0.86, views and comments: 0.75). Heatmap visualization (image4.png) helped identify and mitigate multicollinearity.
- Outliers: Detected via boxplots (e.g., image1.png for duration, image2.png for views). Handled by capping at 1.5 * IQR upper limit.
- Class Balance: Original: 93.7% not verified, 6.3% verified. Post-upsampling: Equal (17,884 each).
- Model Features: Final X includes video_duration_sec, claim_status, author_ban_status, video_view_count, video_share_count, video_download_count, video_comment_count (after encoding).
- Data Source: Loaded via pd.read_csv("tiktok_dataset.csv"); real-world TikTok video data for analysis.

For more details, refer to the project README or Jupyter Notebook.
