import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
import time

class DigitalTwinGraph:
    """
    Extension of the Agentic Graph Reasoning system specifically designed for 
    co-discovering a digital twin identity through interaction.
    """
    
    def __init__(self, base_graph=None):
        """Initialize the Digital Twin graph system."""
        # Initialize the core knowledge graph
        self.graph = nx.DiGraph() if base_graph is None else base_graph.copy()
        
        # Additional properties specific to digital twin representation
        self.interaction_history = []
        self.preference_weights = defaultdict(float)
        self.discovery_state = {}
        self.bridge_nodes = []
        self.confidence_scores = {}
        
        # Initialize with some basic identity facets
        self.identity_facets = [
            "preferences", "behaviors", "knowledge", 
            "personality", "values", "relationships",
            "goals", "experiences", "skills"
        ]
        
        # Add facet nodes to the graph
        for facet in self.identity_facets:
            self.graph.add_node(facet, type="facet", confidence=0.1)
            
        # Track emergence metrics
        self.emergence_metrics = {
            'new_connections': [],
            'bridge_confidence': [],
            'identity_coherence': []
        }
    
    def observe_interaction(self, interaction_data):
        """
        Process an interaction to extract identity information.
        
        Parameters:
        -----------
        interaction_data : dict
            Data about an interaction, including:
            - action: What the user did
            - context: Where/when the action occurred
            - duration: How long the interaction lasted
            - response: How the system or environment responded
        """
        # Record the raw interaction
        self.interaction_history.append({
            'timestamp': time.time(),
            'data': interaction_data
        })
        
        # Extract concepts from the interaction
        concepts = self._extract_concepts_from_interaction(interaction_data)
        
        # Update preference weights based on interaction
        for concept, weight in concepts.items():
            self.preference_weights[concept] += weight
            
            # Ensure concept exists in graph
            if concept not in self.graph:
                # Determine which facet this concept most relates to
                related_facet = self._determine_related_facet(concept)
                
                # Add the concept to the graph
                self.graph.add_node(concept, type="concept", 
                                   weight=weight, confidence=0.1)
                
                # Connect to the appropriate facet
                self.graph.add_edge(related_facet, concept, 
                                   relation="INCLUDES", weight=0.1)
            else:
                # Update the weight of existing concept
                current_weight = self.graph.nodes[concept].get('weight', 0)
                self.graph.nodes[concept]['weight'] = current_weight + weight
                
                # Increase confidence slightly with repeated observations
                current_conf = self.graph.nodes[concept].get('confidence', 0.1)
                self.graph.nodes[concept]['confidence'] = min(current_conf + 0.05, 1.0)
        
        # Create relationships between concepts observed in the same interaction
        concept_nodes = list(concepts.keys())
        for i in range(len(concept_nodes)):
            for j in range(i+1, len(concept_nodes)):
                source = concept_nodes[i]
                target = concept_nodes[j]
                
                # Create or strengthen relationship
                if self.graph.has_edge(source, target):
                    # Strengthen existing edge
                    current_weight = self.graph.edges[source, target].get('weight', 0.1)
                    self.graph.edges[source, target]['weight'] = current_weight + 0.1
                else:
                    # Create new edge with default relation
                    relation = self._infer_relation(source, target, interaction_data)
                    self.graph.add_edge(source, target, relation=relation, weight=0.1)
        
        # Update graph metrics
        self._update_emergence_metrics()
    
    def _extract_concepts_from_interaction(self, interaction_data):
        """
        Extract relevant concepts and their importance weights from an interaction.
        
        This is a simplified implementation - in a real system, this would use 
        more sophisticated NLP and behavioral analysis.
        """
        # Simplified concept extraction
        concepts = {}
        
        # Extract from action
        if 'action' in interaction_data:
            action = interaction_data['action']
            # In a real implementation, this would use NLP to extract concepts
            # Here we'll just use a simple string split as demonstration
            action_words = action.lower().split()
            for word in action_words:
                if len(word) > 3:  # Simple filter for meaningful words
                    concepts[word] = concepts.get(word, 0) + 0.5
        
        # Extract from context
        if 'context' in interaction_data:
            context = interaction_data['context']
            context_words = context.lower().split()
            for word in context_words:
                if len(word) > 3:
                    concepts[word] = concepts.get(word, 0) + 0.3
        
        # Consider duration - longer interactions might indicate stronger interest
        if 'duration' in interaction_data:
            duration = interaction_data['duration']
            # Scale all concept weights by duration factor
            duration_factor = min(duration / 60.0, 2.0)  # Cap at 2x for very long interactions
            for concept in concepts:
                concepts[concept] *= duration_factor
        
        return concepts
    
    def _determine_related_facet(self, concept):
        """Determine which identity facet a concept is most related to."""
        # In a real implementation, this would use semantic analysis
        # For demonstration, we'll use a simple random assignment
        return random.choice(self.identity_facets)
    
    def _infer_relation(self, source, target, interaction_data):
        """Infer the relationship type between two concepts."""
        # Common relationship types in identity networks
        relations = [
            "RELATED_TO", "INFLUENCES", "CONTRASTS_WITH", 
            "PART_OF", "LEADS_TO", "ASSOCIATED_WITH"
        ]
        
        # In a real implementation, this would use context and semantic analysis
        # For demonstration, we'll use a random selection
        return random.choice(relations)
    
    def _update_emergence_metrics(self):
        """Update metrics tracking emergent properties of the digital twin."""
        # Count new connections in this update
        new_connections = sum(1 for edge in self.graph.edges(data=True) 
                             if edge[2].get('weight', 0) <= 0.1)
        self.emergence_metrics['new_connections'].append(new_connections)
        
        # Identify bridge nodes (high betweenness centrality)
        if self.graph.number_of_nodes() > 5:
            try:
                betweenness = nx.betweenness_centrality(self.graph)
                sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
                self.bridge_nodes = [node for node, score in sorted_nodes[:5]]
                
                # Calculate average confidence of bridge nodes
                bridge_conf = np.mean([self.graph.nodes[node].get('confidence', 0) 
                                     for node in self.bridge_nodes])
                self.emergence_metrics['bridge_confidence'].append(bridge_conf)
            except:
                self.emergence_metrics['bridge_confidence'].append(0)
        else:
            self.emergence_metrics['bridge_confidence'].append(0)
        
        # Calculate identity coherence (clustering coefficient as proxy)
        try:
            coherence = nx.average_clustering(self.graph.to_undirected())
            self.emergence_metrics['identity_coherence'].append(coherence)
        except:
            self.emergence_metrics['identity_coherence'].append(0)
    
    def discover_identity_aspects(self):
        """
        Analyze the current graph to discover emergent aspects of the digital twin's identity.
        Returns new discoveries since the last call.
        """
        new_discoveries = {}
        
        # Only perform discovery if graph has enough data
        if self.graph.number_of_nodes() < 10:
            return {"status": "insufficient_data", "discoveries": {}}
        
        # Identify communities (potential identity aspects)
        communities = self._identify_communities()
        
        # For each community, extract key themes
        for i, community in enumerate(communities):
            theme = self._extract_community_theme(community)
            
            # Check if this is a new discovery or significant update
            community_key = f"community_{i}"
            if community_key not in self.discovery_state or self._is_significant_update(
                    self.discovery_state.get(community_key, {}), theme):
                new_discoveries[community_key] = theme
                self.discovery_state[community_key] = theme
        
        # Identify potential contradictions or tensions
        contradictions = self._identify_contradictions()
        if contradictions:
            new_discoveries["contradictions"] = contradictions
        
        # Identify strongest preferences
        top_preferences = self._identify_top_preferences()
        new_discoveries["preferences"] = top_preferences
        
        return {
            "status": "success",
            "discoveries": new_discoveries,
            "bridge_nodes": self.bridge_nodes
        }
    
    def _is_significant_update(self, old_theme, new_theme):
        """Determine if a new theme is significantly different from the old one."""
        # Compare confidence levels
        if abs(new_theme["confidence"] - old_theme.get("confidence", 0)) > 0.2:
            return True
        
        # Compare related concepts
        old_concepts = set(old_theme.get("related_concepts", []))
        new_concepts = set(new_theme["related_concepts"])
        
        # If there's significant difference in the related concepts
        if len(old_concepts.symmetric_difference(new_concepts)) > len(old_concepts) / 2:
            return True
            
        # Otherwise, not a significant update
        return False
    
    def _identify_contradictions(self):
        """Identify potential contradictions or tensions in the identity."""
        contradictions = []
        
        # Look for contradictory relationships
        # In a real implementation, this would use semantic analysis
        # For demonstration, we'll look for nodes that have conflicting edge types
        
        # Check nodes with higher weight (more important to identity)
        important_nodes = [n for n, d in self.graph.nodes(data=True) 
                         if d.get('weight', 0) > 0.5]
        
        for node in important_nodes:
            # Get all edges for this node
            out_edges = list(self.graph.out_edges(node, data=True))
            
            # Check for potentially conflicting relationships
            relations = [edge[2].get('relation', '') for edge in out_edges]
            if 'CONTRASTS_WITH' in relations:
                # Find which nodes contrast with this one
                contrasting_nodes = [edge[1] for edge in out_edges 
                                    if edge[2].get('relation', '') == 'CONTRASTS_WITH']
                
                for contrast_node in contrasting_nodes:
                    contradictions.append({
                        "concept1": node,
                        "concept2": contrast_node,
                        "confidence": min(self.graph.nodes[node].get('confidence', 0),
                                        self.graph.nodes[contrast_node].get('confidence', 0))
                    })
        
        return contradictions
    
    def _identify_top_preferences(self):
        """Identify the strongest preferences based on node weights."""
        # Sort nodes by weight
        weighted_nodes = [(n, d.get('weight', 0)) for n, d in self.graph.nodes(data=True) 
                        if d.get('type', '') == 'concept']
        
        # Sort by weight in descending order
        sorted_nodes = sorted(weighted_nodes, key=lambda x: x[1], reverse=True)
        
        # Return top preferences (up to 5)
        top_count = min(5, len(sorted_nodes))
        top_preferences = [{
            "concept": node,
            "weight": weight,
            "confidence": self.graph.nodes[node].get('confidence', 0)
        } for node, weight in sorted_nodes[:top_count]]
        
        return top_preferences
        
    def visualize_identity_graph(self, figsize=(14, 12), save_path=None):
        """Visualize the current state of the digital twin identity graph."""
        plt.figure(figsize=figsize)
        
        # Use a layout that works well for knowledge graphs
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Create node color map based on node type
        node_colors = []
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'facet':
                node_colors.append('lightblue')
            else:
                # Concepts colored by confidence (darker = higher confidence)
                confidence = self.graph.nodes[node].get('confidence', 0)
                # Interpolate between light red (low confidence) and dark red (high confidence)
                node_colors.append((1.0, 1.0 - confidence * 0.8, 1.0 - confidence * 0.8))
        
        # Create node size map based on weight
        node_sizes = []
        for node in self.graph.nodes():
            weight = self.graph.nodes[node].get('weight', 0.1)
            # Scale node size based on weight (base size + weight factor)
            node_sizes.append(300 + weight * 500)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        
        # Create edge colors based on relation type
        edge_colors = []
        for _, _, data in self.graph.edges(data=True):
            relation = data.get('relation', '')
            # Different colors for different relation types
            if relation == 'INCLUDES':
                edge_colors.append('blue')
            elif relation == 'RELATED_TO':
                edge_colors.append('green')
            elif relation == 'CONTRASTS_WITH':
                edge_colors.append('red')
            else:
                edge_colors.append('gray')
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=1.5, alpha=0.7)
        
        # Draw edge labels (only for important edges)
        important_edges = {(u, v): d['relation'] for u, v, d in self.graph.edges(data=True)
                          if d.get('weight', 0) > 0.3}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=important_edges, font_size=8)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        # Highlight bridge nodes
        if self.bridge_nodes:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=self.bridge_nodes,
                                 node_color='yellow', node_size=[
                                     self.graph.nodes[node].get('weight', 0.1) * 600 + 400
                                     for node in self.bridge_nodes
                                 ])
        
        plt.title("Digital Twin Identity Graph")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_emergence_metrics(self, figsize=(15, 5), save_path=None):
        """Plot the evolution of emergence metrics over interactions."""
        if not self.interaction_history:
            print("No interactions to plot metrics for.")
            return
        
        interactions = list(range(len(self.interaction_history)))
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot new connections
        if self.emergence_metrics['new_connections']:
            axes[0].plot(interactions[:len(self.emergence_metrics['new_connections'])], 
                        self.emergence_metrics['new_connections'], 'b-')
            axes[0].set_title('New Connections per Interaction')
            axes[0].set_xlabel('Interaction')
            axes[0].set_ylabel('Count')
        
        # Plot bridge confidence
        if self.emergence_metrics['bridge_confidence']:
            axes[1].plot(interactions[:len(self.emergence_metrics['bridge_confidence'])], 
                        self.emergence_metrics['bridge_confidence'], 'g-')
            axes[1].set_title('Bridge Node Confidence')
            axes[1].set_xlabel('Interaction')
            axes[1].set_ylabel('Confidence')
        
        # Plot identity coherence
        if self.emergence_metrics['identity_coherence']:
            axes[2].plot(interactions[:len(self.emergence_metrics['identity_coherence'])], 
                        self.emergence_metrics['identity_coherence'], 'r-')
            axes[2].set_title('Identity Coherence')
            axes[2].set_xlabel('Interaction')
            axes[2].set_ylabel('Coherence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()


# Demonstration function
def run_digital_twin_demo(num_interactions=20):
    """
    Run a demonstration of the Digital Twin Graph system.
    
    This simulates a series of user interactions and shows how the digital twin
    identity emerges through the co-discovery process.
    """
    # Initialize the digital twin graph
    twin = DigitalTwinGraph()
    
    # Sample interaction types for simulation
    interaction_types = [
        {"type": "content_engagement", "actions": [
            "read article about AI ethics",
            "watched video on quantum computing",
            "explored virtual art gallery",
            "listened to classical music",
            "played strategy game",
        ]},
        {"type": "creation", "actions": [
            "wrote poem about nature",
            "designed 3D model of futuristic city",
            "composed electronic music track",
            "sketched portrait of friend",
            "built virtual garden simulation",
        ]},
        {"type": "communication", "actions": [
            "discussed philosophy with friend",
            "debated ethical implications of AI",
            "shared scientific article with colleagues",
            "asked question about programming",
            "gave feedback on creative project",
        ]},
        {"type": "environment_change", "actions": [
            "customized virtual workspace with plants",
            "changed color scheme to blue tones",
            "organized digital files by project",
            "added ambient background sounds",
            "adjusted lighting to evening mode",
        ]}
    ]
    
    # Sample contexts
    contexts = [
        "morning relaxation time",
        "focused work session",
        "creative exploration hour",
        "social connection period",
        "learning new skills time",
        "problem-solving session",
        "leisure browsing time",
        "virtual meeting",
        "collaborative project space",
        "contemplative reflection space"
    ]
    
    print(f"Running digital twin simulation with {num_interactions} interactions...")
    
    # Simulate interactions
    discoveries = []
    
    for i in range(num_interactions):
        # Generate a random interaction
        interaction_category = random.choice(interaction_types)
        action = random.choice(interaction_category["actions"])
        context = random.choice(contexts)
        duration = random.randint(5, 120)  # Duration in minutes
        
        # Create interaction data
        interaction = {
            "action": action,
            "context": context,
            "duration": duration,
            "category": interaction_category["type"]
        }
        
        print(f"\nInteraction {i+1}:")
        print(f"  Action: {action}")
        print(f"  Context: {context}")
        print(f"  Duration: {duration} minutes")
        
        # Process the interaction
        twin.observe_interaction(interaction)
        
        # Every few interactions, check for emergent properties
        if (i+1) % 5 == 0 or i == num_interactions - 1:
            print(f"\nAnalyzing identity after {i+1} interactions...")
            discovery = twin.discover_identity_aspects()
            
            if discovery["status"] == "insufficient_data":
                print("  Not enough data yet for meaningful discovery.")
            else:
                discoveries.append(discovery)
                
                # Print new discoveries
                print("  New identity aspects discovered:")
                for key, value in discovery["discoveries"].items():
                    if key == "preferences":
                        print("    Top preferences:")
                        for pref in value:
                            print(f"      - {pref['concept']} (confidence: {pref['confidence']:.2f})")
                    elif key == "contradictions":
                        print("    Potential contradictions:")
                        for contra in value:
                            print(f"      - {contra['concept1']} vs {contra['concept2']}")
                    elif key.startswith("community"):
                        print(f"    Community theme: {value['name']}")
                        if "related_concepts" in value and value["related_concepts"]:
                            print(f"      Related concepts: {', '.join(value['related_concepts'])}")
                
                print("  Bridge nodes (connecting different aspects):")
                for node in discovery["bridge_nodes"]:
                    print(f"    - {node}")
    
    # Final visualization
    print("\nGenerating final visualizations...")
    
    # Visualize the graph
    twin.visualize_identity_graph()
    
    # Plot emergence metrics
    twin.plot_emergence_metrics()
    
    print("\nDigital twin co-discovery complete!")
    print(f"Final graph has {twin.graph.number_of_nodes()} nodes and {twin.graph.number_of_edges()} edges")
    
    # Return the twin object for further exploration
    return twin


if __name__ == "__main__":
    # Run the demonstration
    twin = run_digital_twin_demo(num_interactions=30)
    
    def _identify_communities(self):
        """Identify communities in the graph using a community detection algorithm."""
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()
            
            # Use a community detection algorithm
            communities = list(nx.community.greedy_modularity_communities(undirected))
            return communities
        except:
            # If community detection fails, return empty list
            return []
    
    def _extract_community_theme(self, community):
        """Extract the central theme or concept from a community."""
        if not community:
            return {"name": "unknown", "confidence": 0}
        
        # Get the subgraph for this community
        subgraph = self.graph.subgraph(community)
        
        # Find the node with highest degree as central concept
        central_node = max(subgraph.degree(), key=lambda x: x[1])[0]
        
        # Get related nodes (immediate neighbors)
        neighbors = list(subgraph.neighbors(central_node))
        
        # Calculate the average weight of nodes in the community
        avg_weight = np.mean([subgraph.nodes[node].get('weight', 0) 
                             for node in subgraph.nodes()])
        
        # Calculate average confidence
        avg_confidence = np.mean([subgraph.nodes[node].get('confidence', 0) 
                                for node in subgraph.nodes()])
        
        # Return the theme information
        return {
            "name": central_node,
            "related_concepts": neighbors[:5] if len(neighbors) >= 5 else neighbors,  # Top related concepts
            "weight": avg_weight,
            "confidence": avg_confidence
        }