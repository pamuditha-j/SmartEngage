import './App.css';
import Home from './components/pages/Home';
import Student from './components/pages/Student';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route path='/' element={<Home />} />
          <Route path='/student' element={<Student />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
