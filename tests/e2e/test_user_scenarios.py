# tests/e2e/test_user_scenarios.py - Real user scenario tests
import pytest
import requests
import time
import io
from PIL import Image, ImageDraw
import random

class TestUserScenarios:
    """Test realistic user scenarios"""
    
    def create_realistic_test_image(self, scenario="car"):
        """Create more realistic test images for different scenarios"""
        img = Image.new('RGB', (640, 480), color='white')
        draw = ImageDraw.Draw(img)
        
        if scenario == "car":
            # Draw a simple car shape
            draw.rectangle([100, 200, 400, 300], fill='red', outline='black', width=3)
            draw.rectangle([80, 250, 120, 280], fill='black')  # wheel
            draw.rectangle([380, 250, 420, 280], fill='black')  # wheel
            draw.rectangle([150, 180, 200, 220], fill='lightblue')  # window
            
        elif scenario == "people":
            # Draw simple people shapes
            for i in range(3):
                x = 150 + i * 150
                draw.ellipse([x, 100, x+50, 150], fill='peachpuff')  # head
                draw.rectangle([x+10, 150, x+40, 250], fill='blue')  # body
                draw.rectangle([x+15, 250, x+25, 300], fill='brown')  # leg
                draw.rectangle([x+25, 250, x+35, 300], fill='brown')  # leg
        
        elif scenario == "landscape":
            # Draw simple landscape
            draw.rectangle([0, 300, 640, 480], fill='green')  # ground
            draw.ellipse([200, 50, 300, 150], fill='yellow')  # sun
            draw.polygon([(100, 300), (150, 200), (200, 300)], fill='brown')  # mountain
            draw.polygon([(400, 300), (450, 150), (500, 300)], fill='brown')  # mountain
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)
        return img_bytes
    
    def test_car_identification_scenario(self):
        """Test realistic car identification scenario"""
        base_url = "http://localhost:8000"
        
        print(" Testing car identification scenario...")
        
        # Upload car image
        car_image = self.create_realistic_test_image("car")
        files = {"file": ("red_car.jpg", car_image, "image/jpeg")}
        
        response = requests.post(f"{base_url}/upload", files=files)
        assert response.status_code == 200
        image_id = response.json()["image_id"]
        
        # Ask typical car-related questions
        car_questions = [
            "What vehicle do you see in this image?",
            "What color is the car?",
            "How many wheels can you see?",
            "Is this a sedan or SUV?",
            "What's the overall condition of the vehicle?"
        ]
        
        conversation_id = "car_scenario"
        
        for question in car_questions:
            chat_data = {
                "message": question,
                "image_id": image_id,
                "conversation_id": conversation_id
            }
            
            response = requests.post(f"{base_url}/chat", json=chat_data)
            assert response.status_code == 200
            
            result = response.json()
            print(f"   Q: {question}")
            print(f"   A: {result['response'][:100]}...")
            print(f"   Confidence: {result['confidence']:.0%}")
            print()
        
        print(" Car identification scenario completed!")
    
    def test_people_counting_scenario(self):
        """Test people counting scenario"""
        base_url = "http://localhost:8000"
        
        print(" Testing people counting scenario...")
        
        # Upload people image
        people_image = self.create_realistic_test_image("people")
        files = {"file": ("group_photo.jpg", people_image, "image/jpeg")}
        
        response = requests.post(f"{base_url}/upload", files=files)
        assert response.status_code == 200
        image_id = response.json()["image_id"]
        
        # Ask people-related questions
        people_questions = [
            "How many people are in this image?",
            "What are the people doing?", 
            "Can you describe the people's clothing?",
            "What's the setting or location?",
            "Are the people indoors or outdoors?"
        ]
        
        conversation_id = "people_scenario"
        
        for question in people_questions:
            chat_data = {
                "message": question,
                "image_id": image_id,
                "conversation_id": conversation_id
            }
            
            response = requests.post(f"{base_url}/chat", json=chat_data)
            assert response.status_code == 200
            
            result = response.json()
            print(f"   Q: {question}")
            print(f"   A: {result['response'][:100]}...")
            print(f"   Confidence: {result['confidence']:.0%}")
            print()
        
        print(" People counting scenario completed!")
    
    def test_customer_support_scenario(self):
        """Test customer support use case"""
        base_url = "http://localhost:8000"
        
        print(" Testing customer support scenario...")
        
        # Simulate customer uploading problem image
        problem_image = self.create_realistic_test_image("car")  # Car with issue
        files = {"file": ("car_problem.jpg", problem_image, "image/jpeg")}
        
        response = requests.post(f"{base_url}/upload", files=files)
        assert response.status_code == 200
        image_id = response.json()["image_id"]
        
        # Simulate customer support conversation
        support_conversation = [
            "Hi, I'm having trouble with my car. Can you help?",
            "What do you see in this image of my car?",
            "The car won't start. What could be wrong?",
            "Do you see any obvious damage?",
            "What should I check first?",
            "Thank you for your help!"
        ]
        
        conversation_id = "support_scenario"
        
        for i, message in enumerate(support_conversation):
            chat_data = {
                "message": message,
                "conversation_id": conversation_id
            }
            
            # Include image ID for image-related questions
            if "image" in message.lower() or i == 1:
                chat_data["image_id"] = image_id
            
            response = requests.post(f"{base_url}/chat", json=chat_data)
            assert response.status_code == 200
            
            result = response.json()
            print(f"   Customer: {message}")
            print(f"   AI: {result['response'][:150]}...")
            print(f"   (Confidence: {result['confidence']:.0%})")
            print()
        
        print(" Customer support scenario completed!")
    
    def test_educational_scenario(self):
        """Test educational use case"""
        base_url = "http://localhost:8000"
        
        print(" Testing educational scenario...")
        
        # Upload landscape image for geography lesson
        landscape_image = self.create_realistic_test_image("landscape")
        files = {"file": ("landscape.jpg", landscape_image, "image/jpeg")}
        
        response = requests.post(f"{base_url}/upload", files=files)
        assert response.status_code == 200
        image_id = response.json()["image_id"]
        
        # Educational questions
        educational_questions = [
            "What geographical features do you see in this landscape?",
            "What type of terrain is this?",
            "What's the weather like in this image?",
            "What plants or vegetation can you identify?",
            "How would you describe this ecosystem?"
        ]
        
        conversation_id = "education_scenario"
        
        for question in educational_questions:
            chat_data = {
                "message": question,
                "image_id": image_id,
                "conversation_id": conversation_id
            }
            
            response = requests.post(f"{base_url}/chat", json=chat_data)
            assert response.status_code == 200
            
            result = response.json()
            print(f"   Teacher: {question}")
            print(f"   AI Tutor: {result['response'][:120]}...")
            print(f"   (Confidence: {result['confidence']:.0%})")
            print()
        
        print(" Educational scenario completed!")