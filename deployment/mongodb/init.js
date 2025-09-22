//  deployment/mongodb/init.js - MongoDB initialization
// MongoDB initialization script
db = db.getSiblingDB('multimodal_ai');

// Create collections
db.createCollection('conversations');
db.createCollection('images'); 
db.createCollection('users');
db.createCollection('sessions');

// Create indexes
db.conversations.createIndex({ "conversation_id": 1 });
db.conversations.createIndex({ "timestamp": -1 });
db.images.createIndex({ "image_id": 1 });
db.images.createIndex({ "upload_time": -1 });

// Insert sample data
db.conversations.insertMany([
  {
    conversation_id: "sample_001",
    messages: [
      { role: "user", content: "Hello!" },
      { role: "assistant", content: "Hello! How can I help you today?" }
    ],
    timestamp: new Date(),
    image_count: 0
  }
]);

print(" MongoDB initialization completed");