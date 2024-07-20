# SoundMachine

[Medium Article: A Third Listener - Emotion Detection in Customer Support](https://medium.com/@sensenchen0301/a-third-listener-emotion-detection-in-customer-support-5ee6fa5e963c)

## Background

Introducing **"A Third Listener"**, a machine learning model designed to analyze emotional cues in real-time during customer support interactions. By detecting emotional distress or dissatisfaction, the system can alert representatives or companies to adjust their approach, escalate the issue to experienced personnel, or modify the interaction strategy to better address the customer’s needs.

Resources used in the project are quoted in the Medium article's reference section.

## Want to Test Your Own WAV File?
![image](https://github.com/user-attachments/assets/dc5f4425-62cc-4335-a0c7-370eb94da75c)

There is example wav file in ./third_listener_service/tmp

### Option 1: Online Testing

1. Visit [Inference University](https://inferenceuniversity.ue.r.appspot.com/).
2. Note: We only support WAV file formats.

### Option 2: Local Testing

1. **Clone the Project**
   ```sh
   git clone https://github.com/sensen-c/soundmachine.git
   cd soundmachine/third_listener/third_listener_service
   ```

2.**Install Required Packages**
Ensure you have Python installed and use pip to install the necessary packages:
   ```sh
   pip install -r requirements.txt
```

3.**Run the Service Locally**
   ```sh
   python third_listener/third_listener_service/main.py
```


