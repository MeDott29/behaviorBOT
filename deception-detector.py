import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import random
import datetime
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

@dataclass
class BehavioralCell:
    """Representation of a behavior from the Behavioral Table of Elements"""
    reference_number: str
    symbol: str
    name: str
    confirming_gestures: List[str]
    amplifying_gestures: List[str]
    microphysiological: str
    variables: int
    cultural_prevalence: str
    sexual_propensity: str
    gesture_type: str
    conflicting_behaviors: List[str]
    body_region: str
    deception_rating: float
    deception_timeframe: str

class BehavioralTableOfElements:
    """Implementation of Chase Hughes' Behavioral Table of Elements"""
    
    def __init__(self):
        self.cells: Dict[str, BehavioralCell] = {}
        self._initialize_table()
        
    def _initialize_table(self):
        """Initialize the behavioral table with key elements from Chase Hughes' system"""
        # This is a subset of the full table focusing on high-deception indicators
        
        # Verbal behaviors (high deception indicators)
        self.cells["Res"] = BehavioralCell(
            "114", "Res", "Resume Statement", 
            ["Bg", "Pe"], ["Fr", "Sh"], "None", 1, "U", "U", "Verbal",
            [], "Verbal", 4.0, "D"
        )
        
        self.cells["N-C"] = BehavioralCell(
            "115", "N-C", "Non-Contracting", 
            ["Bg", "Cg"], ["Ag"], "None", 1, "U", "U", "Verbal",
            [], "Verbal", 4.0, "D"
        )
        
        self.cells["Hes"] = BehavioralCell(
            "107", "Hes", "Hesitancy", 
            ["Sw", "Ot"], ["Ye"], "None", 1, "U", "U", "Verbal",
            [], "Verbal", 4.0, "D"
        )
        
        self.cells["Psd"] = BehavioralCell(
            "108", "Psd", "Psychological Distancing", 
            ["Br", "Bg"], ["No"], "None", 1, "U", "U", "Verbal",
            [], "Verbal", 4.0, "D"
        )
        
        # Body language (high deception indicators)
        self.cells["Ss"] = BehavioralCell(
            "38", "Ss", "Single Shoulder Shrug", 
            ["Sw", "Fz"], ["Hl", "Ot"], "None", 1, "U", "U", "Unsure",
            ["Pe"], "Shoulder", 4.0, "D"
        )
        
        self.cells["Fns"] = BehavioralCell(
            "92", "Fns", "Finger-Nose", 
            ["Hu", "Cg"], ["Gm", "Fw"], "None", 1, "U", "U", "Unsure",
            [], "Hands", 4.0, "BD"
        )
        
        self.cells["Df"] = BehavioralCell(
            "82", "Df", "Digital Flexion", 
            ["Pdn", "La"], ["Kc", "Sw"], "None", 1, "U", "U", "Closed",
            ["Ag"], "Hands", 2.0, "BDA"
        )
        
        self.cells["Fw"] = BehavioralCell(
            "91", "Fw", "Foot Withdrawal", 
            ["Cl", "Jp"], ["Kc", "La"], "None", 1, "U", "U", "Closed",
            [], "Feet", 3.5, "BA"
        )
        
        self.cells["Ft"] = BehavioralCell(
            "55", "Ft", "Facial Touch", 
            ["Fns", "Wt"], ["Ip"], "None", 1, "U", "M", "Closed",
            [], "Hands", 3.0, "D"
        )
        
        self.cells["Fz"] = BehavioralCell(
            "54", "Fz", "Freeze", 
            ["Df", "Pr"], ["Gp"], "None", 1, "U", "U", "Closed",
            ["Fr"], "Body", 4.0, "BD"
        )
        
        # Basic indicators (not necessarily deceptive)
        self.cells["Pe"] = BehavioralCell(
            "40", "Pe", "Palm Exposure", 
            ["De", "Bg"], ["Sh", "Ge"], "None", 1, "U", "U", "Open",
            ["Co"], "Hands", 1.0, "DNL"
        )
        
        self.cells["No"] = BehavioralCell(
            "45", "No", "No", 
            ["Br"], ["Df", "Lc"], "None", 1, "U", "U", "Open",
            [], "Head", 1.0, "DNL"
        )
        
        self.cells["Ye"] = BehavioralCell(
            "23", "Ye", "Vertical Headshake", 
            ["Pe"], ["Cg"], "None", 1, "U", "U", "Open",
            ["Br"], "Head", 1.0, "DNL"
        )
        
        self.cells["Br"] = BehavioralCell(
            "26", "Br", "Blink Rate", 
            ["Pe", "Ha"], ["Hd", "Hr"], "None", 5, "U", "U", "Open",
            ["Ct"], "Eyes", 1.5, "BDA"
        )
        
        self.cells["Sw"] = BehavioralCell(
            "36", "Sw", "Swallow", 
            ["Hd", "Pr"], ["Ttc", "Ec"], "Aa", 1, "U", "U", "Closed",
            [], "Neck", 2.5, "BD"
        )
    
    def get_cell(self, symbol: str) -> Optional[BehavioralCell]:
        """Get a behavioral cell by its symbol"""
        return self.cells.get(symbol)
    
    def get_deception_score(self, behaviors: List[str], temp: float = 70.0) -> float:
        """Calculate the deception score for a set of observed behaviors
        
        Args:
            behaviors: List of behavior symbols observed
            temp: Room temperature (affects certain behaviors)
            
        Returns:
            Total deception score
        """
        score = 0.0
        for behavior in behaviors:
            cell = self.get_cell(behavior)
            if cell:
                # Apply temperature adjustment for closed type gestures
                if temp < 69.0 and cell.gesture_type == "Closed":
                    temp_adjustment = (69.0 - temp) // 10
                    adjusted_score = max(0, cell.deception_rating - temp_adjustment)
                    score += adjusted_score
                else:
                    score += cell.deception_rating
        return score
    
    def analyze_behavior_group(self, behaviors: List[str], interviewer_behavior: str = "neutral") -> Dict:
        """Analyze a group of behaviors observed in response to a stimulus
        
        Args:
            behaviors: List of behavior symbols observed
            interviewer_behavior: How the interviewer is behaving (affects interpretation)
            
        Returns:
            Analysis results including deception likelihood
        """
        raw_score = self.get_deception_score(behaviors)
        
        # Adjust for interviewer behavior
        adjusted_score = raw_score
        if interviewer_behavior == "confrontational":
            # Reduce deception scores when interviewer is confrontational
            for behavior in behaviors:
                cell = self.get_cell(behavior)
                if cell and cell.deception_rating == 4.0:
                    adjusted_score -= 2.0
                elif cell and (cell.deception_rating == 3.0 or cell.deception_rating == 3.5):
                    adjusted_score -= 1.0
        
        # Get behavior details
        behaviors_details = []
        for b in behaviors:
            cell = self.get_cell(b)
            if cell:
                behaviors_details.append({
                    "symbol": cell.symbol,
                    "name": cell.name,
                    "deception_rating": cell.deception_rating,
                    "body_region": cell.body_region,
                    "timeframe": cell.deception_timeframe
                })
        
        # Classify deception likelihood
        if adjusted_score >= 12.0:
            deception_likelihood = "High"
        elif adjusted_score >= 8.0:
            deception_likelihood = "Moderate"
        elif adjusted_score >= 4.0:
            deception_likelihood = "Low"
        else:
            deception_likelihood = "Very Low"
            
        return {
            "raw_score": raw_score,
            "adjusted_score": adjusted_score,
            "behaviors": behaviors_details,
            "deception_likelihood": deception_likelihood
        }

class DeceptionAnalysisSystem:
    """System for analyzing and recording deceptive behaviors during interviews"""
    
    def __init__(self):
        self.btoe = BehavioralTableOfElements()
        self.interview_data = []
        self.current_subject = ""
        self.current_session = ""
        
    def start_new_session(self, subject_name: str):
        """Start a new interview session"""
        self.current_subject = subject_name
        self.current_session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interview_data = []
        
    def record_behavior_group(self, question: str, answer: str, behaviors: List[str], 
                              interviewer_behavior: str = "neutral", notes: str = ""):
        """Record a behavior group from an interview interaction"""
        if not self.current_session:
            raise ValueError("No active session. Call start_new_session first.")
            
        analysis = self.btoe.analyze_behavior_group(behaviors, interviewer_behavior)
        
        interaction = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "behaviors": behaviors,
            "interviewer_behavior": interviewer_behavior,
            "deception_score": analysis["adjusted_score"],
            "deception_likelihood": analysis["deception_likelihood"],
            "notes": notes
        }
        
        self.interview_data.append(interaction)
        return analysis
    
    def get_session_summary(self) -> Dict:
        """Get a summary of the current interview session"""
        if not self.interview_data:
            return {"status": "No data recorded"}
        
        total_score = sum(i["deception_score"] for i in self.interview_data)
        avg_score = total_score / len(self.interview_data)
        
        high_deception_responses = [i for i in self.interview_data 
                                    if i["deception_likelihood"] in ["Moderate", "High"]]
        
        most_common_behaviors = {}
        for interaction in self.interview_data:
            for behavior in interaction["behaviors"]:
                if behavior in most_common_behaviors:
                    most_common_behaviors[behavior] += 1
                else:
                    most_common_behaviors[behavior] = 1
        
        top_behaviors = sorted(most_common_behaviors.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "subject": self.current_subject,
            "session_id": self.current_session,
            "interaction_count": len(self.interview_data),
            "average_deception_score": avg_score,
            "high_deception_count": len(high_deception_responses),
            "top_behaviors": top_behaviors
        }
    
    def save_session(self, filename: str = None):
        """Save the current session data to a CSV file"""
        if not self.interview_data:
            return {"status": "No data to save"}
            
        if filename is None:
            filename = f"interview_{self.current_subject}_{self.current_session}.csv"
            
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["timestamp", "question", "answer", "behaviors", 
                          "interviewer_behavior", "deception_score", 
                          "deception_likelihood", "notes"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for interaction in self.interview_data:
                # Convert behaviors list to string for CSV
                interaction_copy = interaction.copy()
                interaction_copy["behaviors"] = ','.join(interaction_copy["behaviors"])
                writer.writerow(interaction_copy)
                
        return {"status": "success", "filename": filename}

class DeceptionDetectorGUI:
    """Graphical interface for the Deception Analysis System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Behavioral Analysis System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        self.system = DeceptionAnalysisSystem()
        self.selected_behaviors = []
        
        # Create a style
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", background="#4a7abc", foreground="black", font=("Helvetica", 10))
        self.style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 10))
        self.style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
        self.style.configure("Result.TLabel", font=("Helvetica", 11), background="#f0f0f0")
        
        # Main frames
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for subject info
        self.top_frame = ttk.Frame(self.main_frame, padding="5")
        self.top_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.top_frame, text="Subject Name:", style="Header.TLabel").pack(side=tk.LEFT, padx=5)
        self.subject_name = ttk.Entry(self.top_frame, width=25)
        self.subject_name.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.top_frame, text="Start New Session", command=self.start_new_session).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.top_frame, text="Save Session", command=self.save_session).pack(side=tk.LEFT, padx=5)
        
        self.session_status = ttk.Label(self.top_frame, text="No active session", font=("Helvetica", 10, "italic"))
        self.session_status.pack(side=tk.RIGHT, padx=10)
        
        # Left frame for behaviors
        self.left_frame = ttk.Frame(self.main_frame, padding="5", width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(self.left_frame, text="Observed Behaviors:", style="Header.TLabel").pack(anchor="w", pady=5)
        
        # Create behavior checkboxes
        self.behavior_vars = {}
        self.create_behavior_checkboxes()
        
        # Interviewer behavior dropdown
        ttk.Label(self.left_frame, text="Interviewer Behavior:", style="Header.TLabel").pack(anchor="w", pady=(10, 5))
        self.interviewer_behavior = tk.StringVar(value="neutral")
        interviewer_options = ttk.Combobox(self.left_frame, textvariable=self.interviewer_behavior)
        interviewer_options['values'] = ('neutral', 'confrontational', 'supportive')
        interviewer_options.pack(fill=tk.X, pady=5)
        
        # Middle frame for question/answer input
        self.middle_frame = ttk.Frame(self.main_frame, padding="5")
        self.middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(self.middle_frame, text="Interview Exchange:", style="Header.TLabel").pack(anchor="w", pady=5)
        
        input_frame = ttk.Frame(self.middle_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Question:").pack(anchor="w")
        self.question_input = scrolledtext.ScrolledText(input_frame, height=3, width=40, wrap=tk.WORD)
        self.question_input.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Answer:").pack(anchor="w")
        self.answer_input = scrolledtext.ScrolledText(input_frame, height=5, width=40, wrap=tk.WORD)
        self.answer_input.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Notes:").pack(anchor="w")
        self.notes_input = scrolledtext.ScrolledText(input_frame, height=3, width=40, wrap=tk.WORD)
        self.notes_input.pack(fill=tk.X, pady=5)
        
        ttk.Button(input_frame, text="Record Interaction", command=self.record_interaction).pack(pady=10)
        
        # Results frame
        self.results_frame = ttk.Frame(self.middle_frame, padding="5")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(self.results_frame, text="Analysis Results:", style="Header.TLabel").pack(anchor="w", pady=5)
        
        self.results_text = scrolledtext.ScrolledText(self.results_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_text.config(state=tk.DISABLED)
        
        # Right frame for summary statistics and visualization
        self.right_frame = ttk.Frame(self.main_frame, padding="5", width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(self.right_frame, text="Session Summary:", style="Header.TLabel").pack(anchor="w", pady=5)
        
        self.summary_text = scrolledtext.ScrolledText(self.right_frame, height=8, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.X, pady=5)
        self.summary_text.config(state=tk.DISABLED)
    
    def update_visualization(self):
        """Update the visualization with session data"""
        # Clear the current figure
        self.ax.clear()
        
        # If we have data, plot it
        if self.system.interview_data:
            # Extract interaction numbers and deception scores
            interactions = list(range(1, len(self.system.interview_data) + 1))
            scores = [interaction["deception_score"] for interaction in self.system.interview_data]
            
            # Create the line plot
            self.ax.plot(interactions, scores, 'o-', color='blue', linewidth=2, markersize=6)
            self.ax.set_xlabel('Interaction Number')
            self.ax.set_ylabel('Deception Score')
            
            # Add a horizontal line at the deception threshold (12.0)
            self.ax.axhline(y=12.0, color='red', linestyle='--', alpha=0.7)
            self.ax.text(1, 12.5, 'Deception Threshold', color='red', fontsize=8)
            
            # Set y-axis limits
            max_score = max(scores) if scores else 0
            self.ax.set_ylim([0, max(20.0, max_score + 2)])
            
            # Add grid
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add title
            self.ax.set_title('Deception Score Trend')
        else:
            # If no data, show a message
            self.ax.text(0.5, 0.5, 'No data available', 
                         horizontalalignment='center', 
                         verticalalignment='center',
                         transform=self.ax.transAxes)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        
        # Refresh the canvas
        self.canvas.draw()
    
    def save_session(self):
        """Save the current session data"""
        if not self.system.current_session:
            messagebox.showerror("Error", "No active session to save")
            return
            
        # Ask for the file location
        filename = tk.filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"interview_{self.system.current_subject}_{self.system.current_session}.csv"
        )
        
        if not filename:
            return  # User cancelled
            
        result = self.system.save_session(filename)
        
        if result["status"] == "success":
            messagebox.showinfo("Success", f"Session data saved to {result['filename']}")
        else:
            messagebox.showerror("Error", f"Failed to save session: {result['status']}")

class AIDeceptionSimulator:
    """Class to simulate deceptive and truthful responses for demonstration"""
    
    def __init__(self, btoe):
        """Initialize with a reference to the Behavioral Table of Elements"""
        self.btoe = btoe
        self.truthful_behaviors = ["Pe", "No", "Ye", "Br"]
        self.deceptive_verbal = ["Res", "N-C", "Hes", "Psd"]
        self.deceptive_nonverbal = ["Ss", "Fns", "Df", "Fw", "Ft", "Fz"]
        self.stress_baseline = ["Sw", "Df"]  # Common stress behaviors
        
    def generate_truthful_response(self, question, stress_level=0.3):
        """Generate a simulated truthful response"""
        behaviors = []
        
        # Add some baseline truthful behaviors
        for behavior in self.truthful_behaviors:
            if random.random() < 0.7:  # 70% chance of each truthful behavior
                behaviors.append(behavior)
        
        # Add some stress behaviors based on stress level
        for behavior in self.stress_baseline:
            if random.random() < stress_level:
                behaviors.append(behavior)
        
        # Generate a plausible answer
        answer = self._generate_answer(question, is_truthful=True)
        
        return {
            "answer": answer,
            "behaviors": behaviors,
            "ground_truth": "truthful"
        }
    
    def generate_deceptive_response(self, question, deception_skill=0.5):
        """Generate a simulated deceptive response"""
        behaviors = []
        
        # Add some baseline stress behaviors
        for behavior in self.stress_baseline:
            if random.random() < 0.8:  # 80% chance of baseline stress
                behaviors.append(behavior)
        
        # Add verbal deception indicators based on deception skill
        # Lower skill = more obvious deception markers
        verbal_chance = 0.8 - (0.5 * deception_skill)
        for behavior in self.deceptive_verbal:
            if random.random() < verbal_chance:
                behaviors.append(behavior)
        
        # Add nonverbal deception indicators
        nonverbal_chance = 0.7 - (0.4 * deception_skill)
        for behavior in self.deceptive_nonverbal:
            if random.random() < nonverbal_chance:
                behaviors.append(behavior)
        
        # Skilled deceivers may include truthful behaviors to appear honest
        if deception_skill > 0.6:
            for behavior in self.truthful_behaviors:
                if random.random() < 0.4:
                    behaviors.append(behavior)
        
        # Generate a plausible deceptive answer
        answer = self._generate_answer(question, is_truthful=False)
        
        return {
            "answer": answer,
            "behaviors": behaviors,
            "ground_truth": "deceptive"
        }
    
    def _generate_answer(self, question, is_truthful=True):
        """Generate a plausible answer to the question"""
        # Common question types and possible responses
        question_lower = question.lower()
        
        # Simple demo responses
        if "name" in question_lower:
            if is_truthful:
                return "My name is John Smith."
            else:
                return "I'm John... John Smith."
        
        elif "where were you" in question_lower:
            if is_truthful:
                return "I was at home watching TV all evening."
            else:
                return "I was at home. I'm a respectable person, I live in a nice neighborhood and I was just watching some shows on television."
        
        elif "know" in question_lower and "victim" in question_lower:
            if is_truthful:
                return "No, I've never met them before."
            else:
                return "I did not have any relationship with that person."
        
        elif "happened" in question_lower:
            if is_truthful:
                return "I saw the argument start around 8pm, then I left because I felt uncomfortable."
            else:
                return "Well, you know, there was some kind of... discussion going on when I arrived, so I just kept my distance."
        
        elif "take" in question_lower or "steal" in question_lower:
            if is_truthful:
                return "No, I didn't take anything from the office."
            else:
                return "I would never do something like that. I've worked at this company for five years with a perfect record."
        
        # Default responses
        if is_truthful:
            return "No, that's not what happened. I was just trying to help resolve the situation."
        else:
            return "Look, I'm an honest person. I volunteer at the community center and have never been in trouble before. There must be some misunderstanding."

def main():
    root = tk.Tk()
    app = DeceptionDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
        
        # Add visualization area
        ttk.Label(self.right_frame, text="Deception Score Trend:", style="Header.TLabel").pack(anchor="w", pady=5)
        
        self.figure, self.ax = plt.subplots(figsize=(4, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize behavior database
        self.interview_log = []
        
    def create_behavior_checkboxes(self):
        """Create checkboxes for the behaviors from BTOE"""
        behavior_frame = ttk.Frame(self.left_frame)
        behavior_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas with scrollbar for the behaviors
        canvas = tk.Canvas(behavior_frame, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(behavior_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Group behaviors by category
        verbal_behaviors = ["Res", "N-C", "Hes", "Psd"]
        body_behaviors = ["Ss", "Fns", "Df", "Fw", "Ft", "Fz"]
        basic_behaviors = ["Pe", "No", "Ye", "Br", "Sw"]
        
        # Create header for verbal behaviors
        ttk.Label(scrollable_frame, text="Verbal Behaviors", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(5,2))
        
        # Create verbal behavior checkboxes
        row = 1
        for behavior in verbal_behaviors:
            cell = self.system.btoe.get_cell(behavior)
            if cell:
                var = tk.BooleanVar()
                self.behavior_vars[behavior] = var
                
                cb = ttk.Checkbutton(
                    scrollable_frame, 
                    text=f"{cell.symbol} - {cell.name} ({cell.deception_rating})",
                    variable=var,
                    command=lambda b=behavior: self.toggle_behavior(b)
                )
                cb.grid(row=row, column=0, sticky="w", padx=5)
                row += 1
        
        # Create header for body behaviors
        ttk.Label(scrollable_frame, text="Body Language", font=("Helvetica", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(10,2))
        row += 1
        
        # Create body behavior checkboxes
        for behavior in body_behaviors:
            cell = self.system.btoe.get_cell(behavior)
            if cell:
                var = tk.BooleanVar()
                self.behavior_vars[behavior] = var
                
                cb = ttk.Checkbutton(
                    scrollable_frame, 
                    text=f"{cell.symbol} - {cell.name} ({cell.deception_rating})",
                    variable=var,
                    command=lambda b=behavior: self.toggle_behavior(b)
                )
                cb.grid(row=row, column=0, sticky="w", padx=5)
                row += 1
        
        # Create header for basic behaviors
        ttk.Label(scrollable_frame, text="Basic Indicators", font=("Helvetica", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(10,2))
        row += 1
        
        # Create basic behavior checkboxes
        for behavior in basic_behaviors:
            cell = self.system.btoe.get_cell(behavior)
            if cell:
                var = tk.BooleanVar()
                self.behavior_vars[behavior] = var
                
                cb = ttk.Checkbutton(
                    scrollable_frame, 
                    text=f"{cell.symbol} - {cell.name} ({cell.deception_rating})",
                    variable=var,
                    command=lambda b=behavior: self.toggle_behavior(b)
                )
                cb.grid(row=row, column=0, sticky="w", padx=5)
                row += 1
        
        # Add a "clear all" button
        ttk.Button(scrollable_frame, text="Clear All", command=self.clear_all_behaviors).grid(
            row=row, column=0, sticky="w", pady=10)
        
    def toggle_behavior(self, behavior):
        """Toggle a behavior in the selected behaviors list"""
        if self.behavior_vars[behavior].get():
            if behavior not in self.selected_behaviors:
                self.selected_behaviors.append(behavior)
        else:
            if behavior in self.selected_behaviors:
                self.selected_behaviors.remove(behavior)
    
    def clear_all_behaviors(self):
        """Clear all selected behaviors"""
        for var in self.behavior_vars.values():
            var.set(False)
        self.selected_behaviors = []
    
    def start_new_session(self):
        """Start a new interview session"""
        subject = self.subject_name.get().strip()
        if not subject:
            messagebox.showerror("Error", "Please enter a subject name")
            return
            
        self.system.start_new_session(subject)
        self.interview_log = []
        self.session_status.config(text=f"Active session: {subject}")
        self.update_summary()
        self.update_visualization()
        
        # Clear inputs
        self.question_input.delete(1.0, tk.END)
        self.answer_input.delete(1.0, tk.END)
        self.notes_input.delete(1.0, tk.END)
        self.clear_all_behaviors()
        
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"New session started for subject: {subject}\n")
        self.results_text.config(state=tk.DISABLED)
    
    def record_interaction(self):
        """Record an interaction with the subject"""
        if not self.system.current_session:
            messagebox.showerror("Error", "No active session. Start a new session first.")
            return
            
        question = self.question_input.get(1.0, tk.END).strip()
        answer = self.answer_input.get(1.0, tk.END).strip()
        notes = self.notes_input.get(1.0, tk.END).strip()
        
        if not question or not answer:
            messagebox.showerror("Error", "Please enter both question and answer")
            return
            
        if not self.selected_behaviors:
            messagebox.showerror("Error", "Please select at least one observed behavior")
            return
            
        # Record the interaction
        analysis = self.system.record_behavior_group(
            question, 
            answer, 
            self.selected_behaviors,
            self.interviewer_behavior.get(),
            notes
        )
        
        # Update the UI
        self.display_analysis_results(analysis)
        self.update_summary()
        self.update_visualization()
        
        # Clear inputs for next interaction
        self.question_input.delete(1.0, tk.END)
        self.answer_input.delete(1.0, tk.END)
        self.notes_input.delete(1.0, tk.END)
        self.clear_all_behaviors()
    
    def display_analysis_results(self, analysis):
        """Display the analysis results in the results text area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, f"ANALYSIS RESULTS\n", "heading")
        self.results_text.insert(tk.END, f"Deception Score: {analysis['adjusted_score']:.1f}\n")
        self.results_text.insert(tk.END, f"Deception Likelihood: {analysis['deception_likelihood']}\n\n")
        
        self.results_text.insert(tk.END, "Observed Behaviors:\n", "subheading")
        for behavior in analysis['behaviors']:
            self.results_text.insert(
                tk.END, 
                f"• {behavior['symbol']} - {behavior['name']} "
                f"(Score: {behavior['deception_rating']}, Region: {behavior['body_region']})\n"
            )
        
        self.results_text.insert(tk.END, "\nInterpretation:\n", "subheading")
        
        if analysis['deception_likelihood'] == "High":
            interpretation = (
                "The subject is showing multiple strong indicators of deception. "
                "There is a high probability of dishonesty in their response. "
                "Note especially the combination of verbal and non-verbal deceptive behaviors."
            )
        elif analysis['deception_likelihood'] == "Moderate":
            interpretation = (
                "The subject is showing some indicators of deception, but the pattern is not definitive. "
                "There may be deception or the subject may be experiencing stress/anxiety without being deceptive. "
                "Further questioning is recommended."
            )
        elif analysis['deception_likelihood'] == "Low":
            interpretation = (
                "The subject is showing few indicators of deception. "
                "Their response appears mostly credible, though some stress or anxiety may be present."
            )
        else:
            interpretation = (
                "The subject is showing minimal indicators of deception. "
                "Their response appears credible based on behavioral analysis."
            )
            
        self.results_text.insert(tk.END, interpretation)
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results_text.insert(tk.END, f"\n\nRecorded: {timestamp}\n")
        
        self.results_text.config(state=tk.DISABLED)
    
    def update_summary(self):
        """Update the session summary information"""
        summary = self.system.get_session_summary()
        
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        if "status" in summary and summary["status"] == "No data recorded":
            self.summary_text.insert(tk.END, "No data recorded in this session.")
        else:
            self.summary_text.insert(tk.END, f"Subject: {summary['subject']}\n")
            self.summary_text.insert(tk.END, f"Total Interactions: {summary['interaction_count']}\n")
            self.summary_text.insert(tk.END, f"Average Deception Score: {summary['average_deception_score']:.2f}\n")
            self.summary_text.insert(tk.END, f"High Deception Responses: {summary['high_deception_count']}\n\n")
            
            self.summary_text.insert(tk.END, "Most Common Behaviors:\n")
            for behavior, count in summary['top_behaviors']:
                cell = self.system.btoe.get_cell(behavior)
                if cell:
                    self.summary_text.insert(tk.END, f"- {cell.symbol} ({cell.name}): {count}\n")
