let currentModel = null;

// Elemente selektieren
const resetButton = document.getElementById('resetButton');
const uploadForm = document.getElementById('uploadForm');
const resultImage = document.getElementById('resultImage');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const uploadFrame = document.getElementById('uploadFrame');
const cameraSection = document.getElementById('cameraSection');
const cameraStatus = document.getElementById('cameraStatus');
const videoFeed = document.getElementById('videoFeed');
const modelFileInput = document.getElementById('modelFileInput');
const confThreshold = document.getElementById('confThreshold');
const maskAlpha = document.getElementById('maskAlpha');
const maskColor = document.getElementById('maskColor');

// Debounce function to prevent too many requests
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
      const later = () => {
          clearTimeout(timeout);
          func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
  };
}

// Update slider value displays in real-time (without API calls)
confThreshold.addEventListener('input', function() {
  this.nextElementSibling.textContent = this.value;
});

maskAlpha.addEventListener('input', function() {
  this.nextElementSibling.textContent = this.value;
});

// Save settings with debounce to prevent freezing
const debouncedSaveSettings = debounce(saveSettings, 1000);

// Save settings
async function saveSettings() {
  const settings = {
      conf_threshold: parseFloat(confThreshold.value),
      mask_alpha: parseFloat(maskAlpha.value),
      mask_color: maskColor.value.toUpperCase()
  };
  
  try {
      const response = await fetch('/update_settings', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify(settings)
      });
      
      const data = await response.json();
      
      if (data.status !== 'success') {
          console.error('Error saving settings:', data.message);
      }
  } catch (error) {
      console.error('Error saving settings:', error);
  }
}

// Model file upload
modelFileInput.addEventListener('change', async function(e) {
    if (e.target.files[0]) {
        await uploadModel(e.target.files[0]);
    }
});

// Upload model to server
async function uploadModel(file) {
    const formData = new FormData();
    formData.append('model', file);
    
    try {
        const response = await fetch('/upload_model', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            window.location.reload();
        } else {
            alert(`Fehler: ${data.message}`);
        }
    } catch (error) {
        console.error('Error uploading model:', error);
        alert('Fehler beim Hochladen des Modells');
    }
}

// Modell laden
async function loadModel(modelIndex) {
    try {
        const response = await fetch('/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model_index: modelIndex })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            currentModel = modelIndex;
            document.getElementById('cameraPlaceholder').style.display = 'none';
            videoFeed.style.display = 'block';
            videoFeed.src = '/video_feed';
            updateCameraStatus();
            // Start regular status updates
            startStatusUpdates();
        } else {
            alert(`Fehler: ${data.message}`);
        }
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Fehler beim Laden des Modells');
    }
}

// Format model information for display
function formatModelInfo(modelInfo) {
    let lines = [];
    
    // Line 1: GPU Status
    if (modelInfo.device === 'GPU') {
        const gpuModel = modelInfo.hw_status.split('\n')[2] || 'NVIDIA GPU';
        const gpuName = gpuModel.replace('GPU: ', '');
        lines.push(`GPU Erkannt! Model: ${gpuName} -- GPU Beschleunigung AKTIV`);
    } else {
        lines.push('Keine GPU Erkannt! -- GPU Beschleunigung INAKTIV');
    }
    
    // Line 2: CPU Status
    const cpuModel = modelInfo.hw_status.split('\n')[1] || 'Intel CPU';
    if (modelInfo.device === 'CPU') {
        lines.push(`CPU Erkannt! Model: ${cpuModel} -- CPU Beschleunigung AKTIV`);
    } else {
        lines.push(`CPU Erkannt! Model: ${cpuModel} -- Beschleunigung INAKTIV`);
    }
    
    // Line 3: Model name and dimensions
    const modelName = modelInfo.hw_status.split('\n')[0] || 'Model';
    lines.push(`${modelName} → Dimensionen: ${modelInfo.input_shape || 'Unknown'}`);
    
    return lines.join('<br>');
}

// Update camera status with model information
async function updateCameraStatus() {
    try {
        const response = await fetch('/get_model_info');
        const modelInfo = await response.json();
        
        // Format and display model info
        cameraStatus.innerHTML = formatModelInfo(modelInfo);
        
        // Style based on status
        if (modelInfo.device === 'GPU') {
            cameraStatus.style.color = '#2ecc71';
        } else if (modelInfo.hw_status?.includes('Error')) {
            cameraStatus.style.color = '#e74c3c';
        } else {
            cameraStatus.style.color = '#3498db';
        }
    } catch (error) {
        console.error('Error getting model info:', error);
    }
}

// Start regular status updates
function startStatusUpdates() {
    // Update immediately
    updateCameraStatus();
    
    // Update every 5 seconds
    if (window.statusInterval) {
        clearInterval(window.statusInterval);
    }
    window.statusInterval = setInterval(updateCameraStatus, 5000);
}

// Modell löschen
async function deleteModel(modelIndex) {
    const confirmDelete = confirm("Möchtest du dieses Modell wirklich löschen?");
    if (!confirmDelete) return;
    
    try {
        const response = await fetch(`/delete_model/${modelIndex}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            window.location.reload();
        } else {
            alert(`Fehler: ${data.message}`);
        }
    } catch (error) {
        console.error('Error deleting model:', error);
        alert('Fehler beim Löschen des Modells');
    }
}

// Modell umbenennen
async function renameModel(modelIndex, nameElement) {
    const currentName = nameElement.textContent;
    const newName = prompt("Neuer Name für das Modell:", currentName);
    
    if (!newName || newName === currentName) return;
    
    try {
        const response = await fetch('/rename_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_index: modelIndex,
                new_name: newName
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            window.location.reload();
        } else {
            alert(`Fehler: ${data.message}`);
        }
    } catch (error) {
        console.error('Error renaming model:', error);
        alert('Fehler beim Umbenennen des Modells');
    }
}

// Cooldown zurücksetzen
resetButton.addEventListener('click', async function() {
    try {
        const response = await fetch('/reset_cooldown', {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log('Cooldown zurückgesetzt!');
        }
    } catch (error) {
        console.error('Error resetting cooldown:', error);
        alert('Fehler beim Zurücksetzen des Cooldowns');
    }
});

// Upload-Formular
uploadForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    if (!currentModel && currentModel !== 0) {
        alert('Bitte zuerst ein Modell auswählen!');
        return;
    }
    
    const formData = new FormData(this);
    
    try {
        const response = await fetch('/detect_image', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            resultImage.src = imageUrl;
            resultImage.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
        } else {
            const data = await response.json();
            alert(`Fehler: ${data.message}`);
        }
    } catch (error) {
        console.error('Error during detection:', error);
        alert('Fehler bei der Objekterkennung');
    }
});

// Tastatur-Shortcuts
document.addEventListener('keydown', function(e) {
    if (e.key >= '1' && e.key <= '9') {
        const modelIndex = parseInt(e.key) - 1;
        const modelItems = document.querySelectorAll('.model-item');
        if (modelIndex < modelItems.length) {
            loadModel(modelIndex);
        }
    }
    if (e.key.toLowerCase() === 'r' && currentModel !== null) {
        resetButton.click();
    }
    if (e.key === 'Escape') {
        alert('Tastatur-Shortcuts:\n\n1-9 - Modell auswählen\nR - Cooldown zurücksetzen\nEsc - Hilfe');
    }
});

// Verbindungsstatus
videoFeed.addEventListener('error', function() {
    cameraStatus.textContent = 'Kamera-Verbindung verloren';
    cameraStatus.style.color = '#e74c3c';
});

document.addEventListener('DOMContentLoaded', function() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
});

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('theme-icon');
  icon.textContent = theme === 'dark' ? '☀️' : '🌙';
}