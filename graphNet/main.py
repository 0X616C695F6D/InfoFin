import sys
import math
import random
import json
import os
import networkx as nx
from collections import defaultdict

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QMessageBox, QToolTip, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
import pyqtgraph as pg

# ------------------------------
# Draggable & Selectable Node Item
# ------------------------------
class DraggableScatterPlotItem(pg.ScatterPlotItem):
    """
    Custom ScatterPlotItem that supports node selection and dragging.
    Stores the original spot data so that positions can be updated during dragging.
    """
    nodeSelectionChanged = pyqtSignal(dict)
    nodesMoved = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self._dragging = False
        self._drag_start_pos = None
        self.selectedPoints = []  # List of currently selected spots
        self.spotsData = []       # Store the original list of spot dictionaries

    def setData(self, *args, **kwargs):
        # Intercept the data passed to setData and store it.
        if args:
            data = args[0]
        elif 'data' in kwargs:
            data = kwargs['data']
        else:
            data = []
        self.spotsData = data.copy()  # make a copy of the spot list
        return super().setData(*args, **kwargs)

    def mousePressEvent(self, event):
        pos = event.pos()
        clicked_points = self.pointsAt(pos)
        modifiers = event.modifiers()
        if clicked_points:
            if modifiers & Qt.ControlModifier:
                for p in clicked_points:
                    if p not in self.selectedPoints:
                        self.selectedPoints.append(p)
                    else:
                        self.selectedPoints.remove(p)
            else:
                self.selectedPoints = clicked_points
            self._dragging = True
            self._drag_start_pos = event.pos()
            self.updateSelectionAppearance()
            selected_metadata = {}
            for p in self.selectedPoints:
                data = p.data()  # data is a dict with keys "id" and "metadata"
                selected_metadata[data['id']] = data['metadata']
            self.nodeSelectionChanged.emit(selected_metadata)
            event.accept()
        else:
            self.selectedPoints = []
            self.updateSelectionAppearance()
            self.nodeSelectionChanged.emit({})
            event.ignore()

    def mouseMoveEvent(self, event):
        if self._dragging and self.selectedPoints:
            delta = event.pos() - self._drag_start_pos
            # For each selected point, update its position in the stored data.
            for p in self.selectedPoints:
                node_id = p.data()['id']
                # Find the corresponding dictionary in spotsData.
                for spot in self.spotsData:
                    if spot['data']['id'] == node_id:
                        oldPos = spot['pos']
                        newPos = (oldPos[0] + delta.x(), oldPos[1] + delta.y())
                        spot['pos'] = newPos
                        break
            # Reapply the updated spotsData.
            self.setData(self.spotsData)
            self._drag_start_pos = event.pos()
            self.nodesMoved.emit()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self.nodesMoved.emit()
        event.accept()

    def updateSelectionAppearance(self):
        """
        Updates each node's appearance. Selected nodes are highlighted in red,
        while non-selected nodes remain black.
        """
        for p in self.points():
            if p in self.selectedPoints:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush(0, 0, 0))


# ------------------------------
# Custom Edge Item with Hover and Click Interaction
# ------------------------------
class EdgeItem(pg.PlotCurveItem):
    edgeClicked = pyqtSignal(dict)
    def __init__(self, x, y, metadata, *args, **kwargs):
        super().__init__(x=x, y=y, *args, **kwargs)
        self.metadata = metadata
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        tooltip_text = (f"Protocol: {self.metadata.get('protocol', 'Unknown')}\n"
                        f"Packets: {self.metadata.get('packets', 0)}")
        QToolTip.showText(event.screenPos(), tooltip_text)
        event.accept()

    def hoverLeaveEvent(self, event):
        QToolTip.hideText()
        event.accept()

    def mousePressEvent(self, event):
        self.edgeClicked.emit(self.metadata)
        event.accept()



# ------------------------------
# Graph Visualization Widget with Navigation Controls
# ------------------------------
class GraphVisualization(pg.GraphicsLayoutWidget):
    """
    Renders a NetworkX graph using pyqtgraph and provides interactive features:
    - Node selection and dragging.
    - Edge interaction (hover and click).
    - Zoom and pan (via built-in mouse interactions).
    - Double-click to center and zoom in on a node.
    - Auto-recenter if nodes drift too far apart.
    """
    nodeSelected = pyqtSignal(dict)  # Emits dict of node_id -> metadata.
    edgeSelected = pyqtSignal(dict)  # Emits edge metadata on click.

    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up the main plot.
        self.plot_item = self.addPlot()
        self.plot_item.setAspectLocked(True)
        self.plot_item.hideAxis('left')
        self.plot_item.hideAxis('bottom')
        grid = pg.GridItem(pen=pg.mkPen('#44475a', width=1, style=Qt.DotLine))
        grid = pg.GridItem(pen=pg.mkPen('#44475a', width=1, style=Qt.DotLine))
        grid.setZValue(-100)  # Ensure the grid is behind other items.
        self.plot_item.vb.addItem(grid)
        # PanMode for easy controls
        self.plot_item.vb.setMouseMode(pg.ViewBox.PanMode)

        # Use custom draggable scatter plot for nodes.
        self.node_scatter = DraggableScatterPlotItem(
            clickable=True,
            size=30,
            pen=pg.mkPen(width=2, color='k'),
            brush=pg.mkBrush(0,0,0,0)
        )
        self.plot_item.addItem(self.node_scatter)
        self.node_scatter.nodeSelectionChanged.connect(self.on_node_selection_changed)
        self.node_scatter.nodesMoved.connect(self.on_nodes_moved)

        self.node_data = {}        # Additional node metadata.
        self.node_positions = {}   # Mapping: node_id -> (x, y).
        self.graph = nx.Graph()    # NetworkX graph instance.
        self.edge_items = []       # List of EdgeItem objects.
        self.edge_node_pairs = []  # List of tuples: (u, v) for each edge.

    def set_graph(self, graph):
        """
        Sets the NetworkX graph, computes a force-directed layout, and stores positions.
        """
        self.graph = graph
        pos = nx.spring_layout(self.graph, seed=42)
        self.node_positions = {node: (coord[0]*100, coord[1]*100) for node, coord in pos.items()}
        self.update_visualization()

    def update_visualization(self):
        """
        Renders nodes and edges using stored positions.
        The edge endpoints are adjusted so they always connect at the node's border.
        Edges are colored based on their protocol.
        """
        # Clear existing nodes and edges.
        self.clear_visualization()

        node_size = 30
        node_radius_pixels = node_size / 2  # 15 pixels

        # Get the size of one pixel in scene units.
        vb = self.plot_item.vb
        pixel_size = vb.viewPixelSize()  # returns a tuple, e.g., (scene_x, scene_y)
        scene_offset = node_radius_pixels * pixel_size[0]  # Use the x-value of the pixel size

        # Define protocol colors.
        protocol_colors = {
            'TCP': (255,255,255),         # Blue
            'UDP': (91,201,79),         # Green
            'ICMP': (255, 0, 0),        # Red
            'HTTP': (255, 165, 0),      # Orange
            'DNS': (128, 0, 128),       # Purple
            'ARP': (0, 255, 255),       # Cyan
            'TLS/SSL': (255, 192, 203), # Pink
            'SSH': (139, 69, 19),       # Brown
            'FTP': (0, 128, 128),       # Teal
            'SMTP': (128, 128, 0),      # Olive
            'POP': (255, 20, 147),      # Deep Pink
            'IMAP': (75, 0, 130),       # Indigo
            'DHCP': (210, 105, 30)      # Chocolate
        }

        # Build node spots (nodes are drawn as black circles).
        spots = []
        self.node_data = {}
        for node, position in self.node_positions.items():
            self.node_data[node] = self.graph.nodes[node]
            spots.append({
                'pos': position,
                'size': node_size,
                'pen': {'color': 'k', 'width': 2},
                'brush': pg.mkBrush(0, 0, 0),  # Black fill for nodes.
                'symbol': 'o',
                'data': {'id': node, 'metadata': self.graph.nodes[node]}
            })
        self.node_scatter.setData(spots)

        # Draw edges.
        self.edge_items = []
        self.edge_node_pairs = []
        for u, v, data in self.graph.edges(data=True):
            if u in self.node_positions and v in self.node_positions:
                pos_u = self.node_positions[u]
                pos_v = self.node_positions[v]
                dx = pos_v[0] - pos_u[0]
                dy = pos_v[1] - pos_u[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    offset_x = (dx / dist) * scene_offset
                    offset_y = (dy / dist) * scene_offset
                else:
                    offset_x = offset_y = 0
                new_pos_u = (pos_u[0] + offset_x, pos_u[1] + offset_y)
                new_pos_v = (pos_v[0] - offset_x, pos_v[1] - offset_y)

                protocol = data.get('protocol', 'Unknown')
                pen_color = protocol_colors.get(protocol, (150, 150, 150))

                edge_line = EdgeItem(
                    x=[new_pos_u[0], new_pos_v[0]],
                    y=[new_pos_u[1], new_pos_v[1]],
                    metadata=data,
                    pen=pg.mkPen(color=pen_color, width=1.5)
                )
                self.plot_item.addItem(edge_line)
                edge_line.edgeClicked.connect(self.on_edge_clicked)
                self.edge_items.append(edge_line)
                self.edge_node_pairs.append((u, v))

        self.plot_item.autoRange()



    def clear_visualization(self):
        """
        Clears all nodes and edges from the plot.
        """
        for edge in self.edge_items:
            self.plot_item.removeItem(edge)
        self.edge_items = []
        self.edge_node_pairs = []
        self.node_scatter.setData([])
        self.node_data = {}

    def on_node_selection_changed(self, selected_nodes):
        self.nodeSelected.emit(selected_nodes)

    def on_nodes_moved(self):
        """
        Called after nodes are dragged. Updates positions, refreshes edges,
        and checks if auto-recenter is needed.
        """
        for p in self.node_scatter.points():
            node_info = p.data()
            pos = p.pos()
            self.node_positions[node_info['id']] = (pos.x(), pos.y())
        self.update_edge_positions()
        self.check_auto_recenter()


    def update_edge_positions(self):
        """
        Updates the positions of all edges when nodes are moved,
        ensuring that edges always connect at the node border.
        """
        import math
        node_size = 30
        node_radius_pixels = node_size / 2  # 15 pixels
        
        vb = self.plot_item.vb
        pixel_size = vb.viewPixelSize()  # returns a tuple (x, y)
        scene_offset = node_radius_pixels * pixel_size[0]
        
        for edge_item, (u, v) in zip(self.edge_items, self.edge_node_pairs):
            if u in self.node_positions and v in self.node_positions:
                pos_u = self.node_positions[u]
                pos_v = self.node_positions[v]
                dx = pos_v[0] - pos_u[0]
                dy = pos_v[1] - pos_u[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    offset_x = (dx / dist) * scene_offset
                    offset_y = (dy / dist) * scene_offset
                else:
                    offset_x = offset_y = 0
                new_pos_u = (pos_u[0] + offset_x, pos_u[1] + offset_y)
                new_pos_v = (pos_v[0] - offset_x, pos_v[1] - offset_y)
                edge_item.setData(x=[new_pos_u[0], new_pos_v[0]], 
                                  y=[new_pos_u[1], new_pos_v[1]])


    def check_auto_recenter(self):
        """
        Checks if the node bounding box exceeds the current view by a threshold.
        If so, automatically re-centers using autoRange().
        """
        xs = [pos[0] for pos in self.node_positions.values()]
        ys = [pos[1] for pos in self.node_positions.values()]
        if not xs or not ys:
            return
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        view_range = self.plot_item.vb.viewRange()  # [[x_min, x_max], [y_min, y_max]]
        view_width = view_range[0][1] - view_range[0][0]
        view_height = view_range[1][1] - view_range[1][0]
        if (max_x - min_x) > 1.5 * view_width or (max_y - min_y) > 1.5 * view_height:
            self.plot_item.autoRange()

    def on_edge_clicked(self, metadata):
        self.edgeSelected.emit(metadata)

    def reset_view(self):
        """
        Resets the view to include all nodes (calls autoRange()).
        """
        self.plot_item.autoRange()

    def mouseDoubleClickEvent(self, event):
        """
        Enables double-clicking on a node to center and zoom in on it.
        If double-clicked near a node (within a 20-pixel threshold), sets view range
        to a fixed window (e.g., 100x100) around that node.
        """
        pos = self.plot_item.vb.mapSceneToView(event.scenePos())
        closest_node = None
        closest_dist = float('inf')
        for p in self.node_scatter.points():
            node_pos = p.pos()
            dist = ((node_pos.x() - pos.x())**2 + (node_pos.y() - pos.y())**2)**0.5
            if dist < closest_dist and dist < 20:
                closest_dist = dist
                node_info = p.data()
                closest_node = node_info['id']
        if closest_node is not None:
            node_pos = self.node_positions[closest_node]
            self.plot_item.setRange(
                xRange=[node_pos[0]-50, node_pos[0]+50],
                yRange=[node_pos[1]-50, node_pos[1]+50],
                padding=0
            )
        else:
            super().mouseDoubleClickEvent(event)

    def create_sample_graph(self):
        """
        Creates and returns a sample NetworkX graph with dummy nodes and edges.
        """
        G = nx.Graph()
        for i in range(10):
            G.add_node(i, type='node', name=f'Node {i}', ip=f"192.168.0.{i}", mac=f"00:11:22:33:44:{i:02x}")
        for _ in range(15):
            u = random.randint(0, 9)
            v = random.randint(0, 9)
            if u != v:
                G.add_edge(u, v, protocol='TCP', packets=random.randint(1, 10))
        return G


# ------------------------------
# FlowGraph Widget
# ------------------------------
class FlowTimelineWidget(pg.GraphicsLayoutWidget):
    """
    A widget that displays a vertical timeline of connections.
    Each connection (edge) is represented by a clickable node arranged in a vertical column.
    The nodes are drawn as black circles with an order number next to them,
    and arrows connect consecutive nodes.
    When a node is clicked, it turns red and emits its metadata.
    """
    connectionClicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create a PlotItem for the timeline.
        self.plot = self.addPlot()
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        # Add a grid to the background.
        self.grid = pg.GridItem(pen=pg.mkPen('#44475a', width=1, style=Qt.DotLine))
        self.grid.setZValue(-100)
        self.plot.addItem(self.grid)
        # Create a ScatterPlotItem for timeline nodes.
        self.scatter = pg.ScatterPlotItem(size=20, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 0))
        self.plot.addItem(self.scatter)
        self.scatter.sigClicked.connect(self.onScatterClicked)
        # We'll store the current spots (timeline nodes) here.
        self.current_spots = []
        # Also store order labels and arrow items.
        self.orderLabels = []
        self.arrows = []

    def updateTimeline(self, edges):
        """
        Given a list of edges (each a tuple: (source, destination, data)),
        sort them by 'start_time' and display a vertical timeline.
        Each timeline node is placed at x = 0.5 with a vertical spacing.
        An order number is displayed next to each node, and arrows connect consecutive nodes.
        """
        # Clear the plot (which removes all items).
        self.plot.clear()
        # Re-add the grid and scatter item.
        self.grid = pg.GridItem(pen=pg.mkPen('#44475a', width=1, style=Qt.DotLine))
        self.grid.setZValue(-100)
        self.plot.addItem(self.grid)
        self.plot.addItem(self.scatter)

        # Clear stored labels and arrows.
        self.orderLabels = []
        self.arrows = []

        # Sort edges by start_time.
        sorted_edges = sorted(edges, key=lambda e: e[2].get('start_time', 0))
        spacing = 50  # Adjust vertical spacing as needed.
        spots = []
        for i, (u, v, data) in enumerate(sorted_edges):
            y = i * spacing
            # Create a timeline node (spot) for the connection.
            spot = {
                'pos': (0.5, y),
                'brush': pg.mkBrush(0, 0, 0),  # Black by default.
                'symbol': 'o',
                'size': 20,
                'data': {'source': u, 'destination': v, **data}
            }
            spots.append(spot)
            # Create a label for the order number.
            label = pg.TextItem(text=str(i+1), color='w', anchor=(1, 0.5))
            label.setPos(0.3, y)
            self.plot.addItem(label)
            self.orderLabels.append(label)
            # If not the last node, draw an arrow from this node to the next expected position.
            if i < len(sorted_edges) - 1:
                # We'll draw an arrow pointing downward.
                arrow = pg.ArrowItem(pos=(0.5, y + spacing/2), angle=90, headLen=10, tipAngle=30, baseAngle=20, brush='w')
                self.plot.addItem(arrow)
                self.arrows.append(arrow)
        self.current_spots = spots
        self.scatter.setData(spots)
        # Set fixed view ranges.
        self.plot.setXRange(0, 1)
        self.plot.setYRange(-spacing/2, spacing * (len(spots)))

    def onScatterClicked(self, scatter, points):
        """
        When a timeline node is clicked, reset all nodes to black,
        set the clicked node to red, and emit its connection metadata.
        """
        # Reset all spots to black.
        for spot in self.current_spots:
            spot['brush'] = pg.mkBrush(0, 0, 0)
        if points:
            clicked_data = points[0].data()
            # Find the matching spot based on source and destination.
            for spot in self.current_spots:
                if (spot['data'].get('source') == clicked_data.get('source') and
                    spot['data'].get('destination') == clicked_data.get('destination')):
                    spot['brush'] = pg.mkBrush('r')
                    self.connectionClicked.emit(spot['data'])
                    break
        self.scatter.setData(self.current_spots)



# ------------------------------
# Main Application Window
# ------------------------------
class MainWindow(QMainWindow):
    """
    The main window integrates:
    - A graph visualization canvas with interactive controls.
    - A side panel for displaying node/edge metadata.
    - A status bar for messages.
    - Drag & drop functionality for JSON import.
    
    It wires together JSON parsing, graph updating, and interactive features.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network Traffic Visualizer")
        self.setMinimumSize(1024, 768)
        self._center_window()
        self._init_ui()
        self.setAcceptDrops(True)
        self.statusBar().showMessage("Ready - Drag and drop a Wireshark JSON file to visualize network")
        self._create_sample_graph()

    def _center_window(self):
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

    def _init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        left_layout = QVBoxLayout()
        
        # Left panel: Graph visualization and control bar.
        self.graph_view = GraphVisualization()
        self.graph_view.setBackground('#282a36')
        control_bar = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.graph_view.reset_view)
        control_bar.addWidget(reset_btn)
        control_bar.addStretch()
        left_layout.addLayout(control_bar)
        left_layout.addWidget(self.graph_view, 1)
        
        # Right panel: Vertical layout for metadata and the flow timeline.
        right_layout = QVBoxLayout()
        self.metadata_panel = QLabel("Detailed node & edge information.")
        self.metadata_panel.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.metadata_panel.setWordWrap(True)
        self.metadata_panel.setStyleSheet("background-color: #21222c; color: #f8f8f2; padding: 10px;")
        right_layout.addWidget(self.metadata_panel, 2)  # For example, 2/3 of the space.
        
        self.flow_timeline = FlowTimelineWidget()
        self.flow_timeline.setBackground('#21222c')
        right_layout.addWidget(self.flow_timeline, 1)  # 1/3 of the space.
        
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)
        main_widget.setStyleSheet("background-color: #21222c; color: #f8f8f2;")
        self.setCentralWidget(main_widget)
        
        # Connect signals.
        self.graph_view.nodeSelected.connect(self.update_metadata_panel_for_node)
        self.graph_view.edgeSelected.connect(self.update_metadata_panel_for_edge)
        self.flow_timeline.connectionClicked.connect(self.update_metadata_panel_for_edge)


    def _create_sample_graph(self):
        sample_graph = self.graph_view.create_sample_graph()
        self.graph_view.set_graph(sample_graph)
        self.statusBar().showMessage("Displaying sample graph - Drop a JSON file to load real data")
        edges_list = [(u, v, data) for u, v, data in sample_graph.edges(data=True)]
        self.flow_timeline.updateTimeline(edges_list)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """
        Accepts drag events if a file with a .json extension is detected.
        """
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.json'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """
        Processes the first dropped JSON file.
        """
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self._process_json_file(file_path)
            break

    def _process_json_file(self, file_path):
        """
        Validates and parses the dropped JSON file, updates the graph,
        and refreshes the UI. Error messages are shown if needed.
        """
        self.statusBar().showMessage(f"Processing file: {os.path.basename(file_path)}")
        if not os.path.exists(file_path):
            self._show_error(f"File not found: {file_path}")
            return
        if not file_path.lower().endswith('.json'):
            self._show_error(f"Not a JSON file: {file_path}")
            return
        try:
            parsed_graph = load_and_parse_wireshark_file(file_path)
            self.graph_view.set_graph(parsed_graph)
            total_nodes = parsed_graph.number_of_nodes()
            unique_protocols = {data.get('protocol') for _, _, data in parsed_graph.edges(data=True)}
            status_message = (f"Graph updated: Total nodes: {total_nodes}, "
                              f"Unique protocols: {len(unique_protocols)}")
            self.statusBar().showMessage(status_message)
            edges_list = [(u, v, data) for u, v, data in parsed_graph.edges(data=True)]
            self.flow_timeline.updateTimeline(edges_list)
        except Exception as e:
            self._show_error(str(e))

    def _show_error(self, message):
        self.statusBar().showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Error", message)

    def update_metadata_panel_for_node(self, node_metadata):
        """
        Updates the side panel with metadata for selected node(s).
        """
        if not node_metadata:
            self.metadata_panel.setText("No node selected.")
            return

    def update_metadata_panel_for_node(self, node_metadata):
        """
        Update the side panel with structured metadata for the selected node(s).
        If multiple nodes are selected, display a table for each.
        """
        if not node_metadata:
            self.metadata_panel.setText("No node selected.")
            return

        html = "<html><body>"
        for node, meta in node_metadata.items():
            html += f"<h3>Node: {node}</h3>"
            html += format_metadata_as_table(meta)
        html += "</body></html>"
        self.metadata_panel.setText(html)

    def update_metadata_panel_for_edge(self, edge_metadata):
        """
        Updates the side panel with structured (HTML table) metadata for the selected edge.
        """
        if not edge_metadata:
            self.metadata_panel.setText("No edge selected.")
            return
        # Format the metadata as an HTML table.
        html = format_edge_metadata_as_table(edge_metadata)
        self.metadata_panel.setText(html)


# ------------------------------
# JSON Parsing Functions
# ------------------------------
def parse_wireshark_json(json_data):
    """
    Parses Wireshark JSON export data to create a directed NetworkX graph.
    Extracts fields for nodes (IP addresses) and aggregates edge data.
    """
    G = nx.DiGraph()
    edge_data = defaultdict(list)
    
    protocol_mapping = {
        'tcp': 'TCP', 'udp': 'UDP', 'icmp': 'ICMP',
        'http': 'HTTP', 'dns': 'DNS', 'arp': 'ARP',
        'tls': 'TLS/SSL', 'ssh': 'SSH', 'ftp': 'FTP',
        'smtp': 'SMTP', 'pop': 'POP', 'imap': 'IMAP', 'dhcp': 'DHCP'
    }
    
    for packet_index, packet in enumerate(json_data):
        try:
            layers = packet.get('_source', {}).get('layers', {})
            ip_layer = layers.get('ip', {})
            src_ip = ip_layer.get('ip.src', 'Unknown')
            dst_ip = ip_layer.get('ip.dst', 'Unknown')
            if src_ip == 'Unknown' or dst_ip == 'Unknown':
                continue
            eth_layer = layers.get('eth', {})
            eth_src = eth_layer.get('eth.src', 'Unknown')
            eth_dst = eth_layer.get('eth.dst', 'Unknown')
            frame_layer = layers.get('frame', {})
            time_epoch = float(frame_layer.get('frame.time_epoch', 0))
            frame_len = int(frame_layer.get('frame.len', 0))
            
            protocol = "Unknown"
            for layer_key in layers:
                if layer_key in protocol_mapping:
                    protocol = protocol_mapping[layer_key]
                    break
            if protocol == "Unknown" and 'frame.protocols' in frame_layer:
                protocols = frame_layer['frame.protocols'].split(':')
                for p in reversed(protocols):
                    if p and p in protocol_mapping:
                        protocol = protocol_mapping[p]
                        break
            
            src_port = None
            dst_port = None
            tcp_flags = None
            tcp_layer = layers.get('tcp', {})
            if tcp_layer:
                src_port = tcp_layer.get('tcp.srcport', None)
                dst_port = tcp_layer.get('tcp.dstport', None)
                if 'tcp.flags' in tcp_layer:
                    tcp_flags = tcp_layer['tcp.flags']
            if src_port is None and 'udp' in layers:
                udp_layer = layers['udp']
                src_port = udp_layer.get('udp.srcport', None)
                dst_port = udp_layer.get('udp.dstport', None)
            if isinstance(src_port, str) and src_port.isdigit():
                src_port = int(src_port)
            if isinstance(dst_port, str) and dst_port.isdigit():
                dst_port = int(dst_port)
            
            if src_ip not in G:
                G.add_node(src_ip, type='host', mac=eth_src, ip=src_ip)
            if dst_ip not in G:
                G.add_node(dst_ip, type='host', mac=eth_dst, ip=dst_ip)
            
            edge_key = (src_ip, dst_ip, protocol)
            packet_data = {
                'packet_index': packet_index,
                'time': time_epoch,
                'length': frame_len,
                'src_mac': eth_src,
                'dst_mac': eth_dst,
                'protocol': protocol,
                'src_port': src_port,
                'dst_port': dst_port,
                'tcp_flags': tcp_flags
            }
            edge_data[edge_key].append(packet_data)
        except Exception as e:
            print(f"Error processing packet {packet_index}: {str(e)}")
            continue
    
    for (src, dst, protocol), packets in edge_data.items():
        total_packets = len(packets)
        total_bytes = sum(p['length'] for p in packets)
        first_packet_time = min(p['time'] for p in packets)
        last_packet_time = max(p['time'] for p in packets)
        duration = last_packet_time - first_packet_time if total_packets > 1 else 0
        src_port = packets[0]['src_port']
        dst_port = packets[0]['dst_port']
        G.add_edge(src, dst, 
                   protocol=protocol, packets=total_packets, bytes=total_bytes,
                   start_time=first_packet_time, end_time=last_packet_time,
                   duration=duration, src_port=src_port, dst_port=dst_port)
    return G

def load_and_parse_wireshark_file(file_path):
    """
    Loads a Wireshark JSON export file, verifies its format, and parses it into a NetworkX graph.
    """
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        if not isinstance(json_data, list) or not json_data:
            raise ValueError("JSON data is not in expected Wireshark format (list of packets)")
        if '_source' not in json_data[0] or 'layers' not in json_data[0].get('_source', {}):
            raise ValueError("JSON data does not appear to be a Wireshark export (missing '_source.layers')")
        return parse_wireshark_json(json_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing Wireshark JSON file: {str(e)}")


# ------------------------------
# HTML Structured Output
# ------------------------------
def format_edge_metadata_as_table(metadata):
    """
    Convert a dictionary of edge metadata into an HTML table.
    If a value is a list or tuple, it is joined with line breaks.
    """
    html = "<html><body><table border='1' cellspacing='0' cellpadding='3'>"
    for key, value in metadata.items():
        # If the value is a list or tuple, join its items with line breaks.
        if isinstance(value, (list, tuple)):
            value = "<br>".join(str(v) for v in value)
        html += f"<tr><th align='left'>{key}</th><td align='left'>{value}</td></tr>"
    html += "</table></body></html>"
    return html

def format_metadata_as_table(metadata, skip_keys=None):
    """
    Converts a metadata dictionary into an HTML table.

    Parameters:
        metadata (dict): The metadata to format.
        skip_keys (list): Keys to omit from the table.

    Returns:
        str: HTML string representing the table.
    """
    if skip_keys is None:
        skip_keys = []
    html = "<html><body><table border='1' cellspacing='0' cellpadding='3'>"
    for key, value in metadata.items():
        if key in skip_keys:
            continue
        # If value is a list or tuple, join its elements on separate lines.
        if isinstance(value, (list, tuple)):
            value = "<br>".join(str(v) for v in value)
        html += f"<tr><th align='left'>{key}</th><td align='left'>{value}</td></tr>"
    html += "</table></body></html>"
    return html

# ------------------------------
# Application Entry Point
# ------------------------------
def main():
    """
    Entry point of the application.
    
    Testing Plan Summary:
    - **Graph Rendering:** On startup, a sample graph is displayed.
    - **Drag & Drop:** Drag a valid Wireshark JSON file onto the window to trigger parsing and graph update.
    - **Node Interaction:** Click on a node to select and view metadata in the side panel; use Ctrl+Click for multi-selection.
    - **Node Dragging:** Drag selected nodes and observe connected edges update in real time.
    - **Double-Click:** Double-click near a node to center and zoom in on it.
    - **Edge Interaction:** Hover over an edge to see a tooltip; click an edge to display detailed metadata.
    - **Zoom & Pan:** Use the mouse wheel to zoom and click-drag (pan) the graph.
    - **Reset View:** Click the "Reset View" button to recenter and rescale the graph.
    - **Error Handling:** Drop a non-JSON file or malformed JSON to trigger error messages.
    
    Each functional block is integrated into the main window, ensuring end-to-end testing.
    """
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
