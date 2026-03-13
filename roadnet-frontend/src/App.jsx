import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import MapView from "./pages/MapView";
import Tickets from "./pages/Tickets";
import Analytics from "./pages/Analytics";
import Departments from "./pages/Departments";
import AIInspection from "./pages/AIInspection";

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/map" element={<MapView />} />
        <Route path="/tickets" element={<Tickets />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/departments" element={<Departments />} />
        <Route path="/inspection" element={<AIInspection />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

