import React from 'react';
import '../App.css';
import { Button } from './Button';
import './LandingSection.css';

function LandingSection() {
  return (
    <div className='parent-container'>
      <video src='/videos/landing_video.mp4' autoPlay loop muted />
      <div className='hero-container'>
        
        <h1 className='landing-title'>SmartEngage</h1>
        <p>With <b>SmartEngage</b>, the classroom becomes a hub of engagement and interactive learning. Educators can inspire and captivate, while students are empowered to take an active role in their education, fostering a community of motivated learners and innovative educators.
         
          Discover how <b>SmartEngage</b> is redefining the boundaries of engagement in education. Join us in creating a more connected, interactive, and effective learning experience for educators and students alike.</p>
        <div className='hero-btns'>
          <Button
            className='btns'
            buttonStyle='btn--outline'
            buttonSize='btn--large'
          >
            LOG IN AS A STUDENT
          </Button>
          <Button
            className='btns'
            buttonStyle='btn--outline'
            buttonSize='btn--large'
            onClick={console.log('hey')}
          >
            LOG IN AS A TUTOR 
          </Button>
        </div>
      </div>
    </div>
  );
}

export default LandingSection;