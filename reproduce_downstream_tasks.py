
import sys
sys.path.append('XHSOpt')
# reproduce_downstream_tasks.py

import logging
from xhs_engagement_optimizer_v2 import EngagementPredictorV2, load_posts_from_database, XHSPost

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to reproduce and demonstrate the two downstream tasks.
    """
    logger.info("--- Starting Downstream Task Reproduction ---")

    # --- Setup ---
    CHECKPOINT_PATH = "moco_True_linkpred_True/current.pth"
    DB_PATH = "xhs_data.db"
    DEVICE = "cpu"
    
    # 1. Initialize the EngagementPredictorV2
    # This class handles model loading, graph processing, and prediction.
    try:
        predictor = EngagementPredictorV2(
            checkpoint_path=CHECKPOINT_PATH,
            device=DEVICE,
            db_path=DB_PATH
        )
        logger.info("✅ EngagementPredictorV2 initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize EngagementPredictorV2: {e}")
        return

    # 2. Load sample posts from the database for demonstration
    # We'll use the same posts from the optimization results for comparison.
    post_ids_to_test = [1, 2, 3, 4, 5]
    posts = load_posts_from_database(DB_PATH, post_ids_to_test)
    if not posts:
        logger.error("❌ Could not load posts from the database.")
        return

    # --- Task 1: Engagement Prediction ---
    logger.info("\n--- Task 1: Engagement Prediction ---")
    for post in posts:
        try:
            score, quality = predictor.predict_engagement(post.content)
            logger.info(f"Post: '{post.content[:40]}...'")
            logger.info(f"  📈 Predicted Engagement Score: {score:.4f}")
            logger.info(f"  🔧 Feature Quality: {quality:.4f}")
        except Exception as e:
            logger.error(f"  ❌ Failed to predict engagement for post: {e}")

    # --- Task 2: Emoji Suggestion ---
    logger.info("\n--- Task 2: Emoji Suggestion ---")
    for post in posts:
        try:
            suggestions = predictor.generate_emoji_suggestions(post.content, top_k=5)
            logger.info(f"Post: '{post.content[:40]}...'")
            logger.info(f"  🎭 Suggested Emojis: {suggestions}")
        except Exception as e:
            logger.error(f"  ❌ Failed to suggest emojis for post: {e}")

    logger.info("\n--- ✅ All tasks completed successfully! ---")

if __name__ == "__main__":
    main()
