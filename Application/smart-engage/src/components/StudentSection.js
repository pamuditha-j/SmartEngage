import React, { useState,useEffect } from "react";
import axios from "axios";
import '../App.css';
import './StudentSection.css';

function StudentSection() {
  const [profileData, setProfileData] = useState(null)

  useEffect(() => {
    const interval = setInterval(async () => {
    axios({
      method: "GET",
      url:"http://127.0.0.1:5001/predictions",
    })
    .then((response) => {
      const res =response.data
      setProfileData(({
        face_angle_horizontal:res.face_angle_horizontal,
        face_angle_vertical:res.face_angle_vertical,
        eye_gaze_horizontal:res.eye_gaze_horizontal,
        eye_gaze_vertical:res.eye_gaze_vertical,
        eye_gaze_text:res.eye_gaze_text,
        affective_state_text:res.affective_state_text,
        basic_emotion_text:res.basic_emotion_text
      }))
        
    }).catch((error) => {
      if (error.response) {
        console.log(error.response)
        console.log(error.response.status)
        console.log(error.response.headers)
        }
    })
  }, 2000);
  return () => clearInterval(interval);
  }, []);

  return (
    <div  style={{height: "100vh", background: "url('/images/background.jpg') center center/cover no-repeat"}}>
        <div>
          {/* <Image/> */}
            <center>
              <h1  className='page-title'>Engagement Detection System</h1>
              <img
              src="http://localhost:5001/"
              alt="Video"
              style={{marginTop:"20px", height:"30%", width:"40%", borderRadius: "20px", border:"2px solid #f8f8f8"}}
              />
              {profileData && <div>
                  <p> Head Pose Horizontal Displacement: {profileData.face_angle_horizontal}</p>
                  <p> Head Pose Vertical Displacement: {profileData.face_angle_vertical}</p>
                  <p> Eye Gaze Horizontal Displacement: {profileData.eye_gaze_horizontal}</p>
                  <p> Eye Gaze Vertical Displacement: {profileData.eye_gaze_vertical}</p>
                  <p> Eye Gaze Prediction: {profileData.eye_gaze_text}</p>
                  <p> Affective State Prediction: {profileData.affective_state_text}</p>
                  <p> Basic Emotion Prediction: {profileData.basic_emotion_text}</p>
                </div>
              }
              
            </center>
        </div>
    </div>
  );
}

export default StudentSection;