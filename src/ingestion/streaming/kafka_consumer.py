"""
Kafka Consumer for Real-Time HIV Data
======================================

Process real-time HIV data streams using Apache Kafka.
"""

import json
from typing import Callable, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import config_manager

logger = get_logger(__name__)


class HIVDataStreamConsumer:
    """
    Consume and process real-time HIV data from Kafka.
    """
    
    def __init__(self, topics: list[str], group_id: str = "hiv-analytics"):
        """
        Initialize Kafka consumer.
        
        Args:
            topics: List of Kafka topics to subscribe to
            group_id: Consumer group ID
        """
        self.topics = topics
        self.group_id = group_id
        
        # Get Kafka configuration
        config = config_manager.get_data_sources_config()
        kafka_config = config.get("streaming", {})
        
        self.brokers = config_manager.get_kafka_brokers()
        
        # Initialize consumer
        try:
            self.consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=self.brokers,
                group_id=group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            )
            
            logger.info(f"Kafka consumer initialized for topics: {topics}")
        
        except KafkaError as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def consume(
        self,
        process_func: Callable[[dict], None],
        max_messages: Optional[int] = None,
    ):
        """
        Consume messages from Kafka topics.
        
        Args:
            process_func: Function to process each message
            max_messages: Maximum number of messages to consume (None = infinite)
        """
        logger.info(f"Starting to consume messages from {self.topics}")
        
        message_count = 0
        
        try:
            for message in self.consumer:
                try:
                    # Process message
                    data = message.value
                    process_func(data)
                    
                    message_count += 1
                    
                    if message_count % 100 == 0:
                        logger.info(f"Processed {message_count} messages")
                    
                    # Check if reached max messages
                    if max_messages and message_count >= max_messages:
                        logger.info(f"Reached maximum of {max_messages} messages")
                        break
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    logger.error(f"Message: {message.value}")
        
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        
        finally:
            self.close()
            logger.info(f"Total messages processed: {message_count}")
    
    def close(self):
        """Close the consumer."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")


class DiagnosisStreamProcessor:
    """
    Process new HIV diagnosis events.
    """
    
    def __init__(self):
        """Initialize processor."""
        self.diagnoses = []
        logger.info("Diagnosis stream processor initialized")
    
    def process(self, data: dict):
        """
        Process a diagnosis event.
        
        Args:
            data: Diagnosis data
        """
        # Validate required fields
        required_fields = ["patient_id", "diagnosis_date", "age", "gender"]
        
        if not all(field in data for field in required_fields):
            logger.warning(f"Missing required fields in diagnosis data: {data}")
            return
        
        # Store diagnosis
        self.diagnoses.append(data)
        
        # Log alert for high-risk cases
        if data.get("cd4_count", 500) < 200:
            logger.warning(
                f"ALERT: Low CD4 count ({data['cd4_count']}) for patient {data['patient_id']}"
            )
        
        # Check for late diagnosis
        who_stage = data.get("who_clinical_stage", "")
        if who_stage in ["Stage 3", "Stage 4"]:
            logger.warning(
                f"ALERT: Late-stage diagnosis ({who_stage}) for patient {data['patient_id']}"
            )
    
    def get_recent_diagnoses(self, n: int = 10) -> pd.DataFrame:
        """
        Get recent diagnoses.
        
        Args:
            n: Number of recent diagnoses to return
        
        Returns:
            DataFrame with recent diagnoses
        """
        if not self.diagnoses:
            return pd.DataFrame()
        
        return pd.DataFrame(self.diagnoses[-n:])


class LabResultStreamProcessor:
    """
    Process lab result events.
    """
    
    def __init__(self):
        """Initialize processor."""
        self.lab_results = []
        logger.info("Lab result stream processor initialized")
    
    def process(self, data: dict):
        """
        Process a lab result event.
        
        Args:
            data: Lab result data
        """
        # Validate required fields
        required_fields = ["patient_id", "test_date", "cd4_count", "viral_load"]
        
        if not all(field in data for field in required_fields):
            logger.warning(f"Missing required fields in lab result: {data}")
            return
        
        # Store result
        self.lab_results.append(data)
        
        # Check for viral suppression
        viral_load = data.get("viral_load", 0)
        if viral_load < 200:
            logger.info(
                f"✅ Viral suppression achieved for patient {data['patient_id']} (VL: {viral_load})"
            )
        elif viral_load > 1000:
            logger.warning(
                f"⚠️ Elevated viral load for patient {data['patient_id']}: {viral_load}"
            )
        
        # Check CD4 count
        cd4_count = data.get("cd4_count", 0)
        if cd4_count < 200:
            logger.warning(
                f"⚠️ Low CD4 count for patient {data['patient_id']}: {cd4_count}"
            )
    
    def calculate_realtime_metrics(self) -> dict:
        """
        Calculate real-time metrics from recent lab results.
        
        Returns:
            Dictionary with metrics
        """
        if not self.lab_results:
            return {}
        
        df = pd.DataFrame(self.lab_results)
        
        metrics = {
            "total_tests": len(df),
            "avg_viral_load": df["viral_load"].mean(),
            "avg_cd4_count": df["cd4_count"].mean(),
            "suppression_rate": (df["viral_load"] < 200).mean() * 100,
            "low_cd4_rate": (df["cd4_count"] < 200).mean() * 100,
        }
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Note: This requires a running Kafka cluster
    
    # Initialize processor
    diagnosis_processor = DiagnosisStreamProcessor()
    
    # Initialize consumer
    try:
        consumer = HIVDataStreamConsumer(
            topics=["hiv.diagnoses.new"],
            group_id="hiv-analytics-test",
        )
        
        # Consume messages
        consumer.consume(
            process_func=diagnosis_processor.process,
            max_messages=100,
        )
        
        # Display results
        recent_diagnoses = diagnosis_processor.get_recent_diagnoses()
        print("\nRecent Diagnoses:")
        print(recent_diagnoses)
    
    except KafkaError as e:
        logger.error(f"Kafka error: {e}")
        print("\n⚠️ Note: This example requires a running Kafka cluster")
        print("To run the streaming pipeline:")
        print("1. Start Kafka: docker-compose up kafka")
        print("2. Run this script")

