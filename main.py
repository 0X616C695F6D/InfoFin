import sys
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
    Custom ScatterPlotItem to enable node selection (with multi-select via Ctrl+Click)
    and dragging. Emits a signal when selection changes or nodes are moved.
    """
    # Emits a dict mapping node ID to its metadata when selection changes.
    nodeSelectionChanged = pyqtSignal(dict)
    # Emits when one or more nodes have been moved.
    nodesMoved = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self._dragging = False
        self._drag_start_pos = None
        self.selectedPoints = []  # List of currently selected spots

    def mousePressEvent(self, event):
        pos = event.pos()  # Position in the item's coordinates
        clicked_points = self.pointsAt(pos)
        modifiers = event.modifiers()
        if clicked_points:
            if modifiers & Qt.ControlModifier:
                # Toggle selection: add if not selected, remove if already selected.
                for p in clicked_points:
                    if p in self.selectedPoints:
                        self.selectedPoints.remove(p)
                    else:
                        self.selectedPoints.append(p)
            else:
                # Clear previous selection and select the clicked point(s).
                self.selectedPoints = clicked_points
            self._dragging = True
            self._drag_start_pos = event.pos()
            self.updateSelectionAppearance()
            # Emit metadata for the selected nodes.
            selected_metadata = {}
            for p in self.selectedPoints:
                data = p.data()  # data is a dict with keys "id" and "metadata"
                selected_metadata[data['id']] = data['metadata']
            self.nodeSelectionChanged.emit(selected_metadata)
            event.accept()
        else:
            # Clicked on blank space; clear selection.
            self.selectedPoints = []
            self.updateSelectionAppearance()
            self.nodeSelectionChanged.emit({})
            event.ignore()

    def mouseMoveEvent(self, event):
        if self._dragging and self.selectedPoints:
            delta = event.pos() - self._drag_start_pos
            for p in self.selectedPoints:
                newPos = p.pos() + delta
                p.setPos(newPos)
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
        Updates each node's appearance. Selected nodes are highlighted in red.
        """
        for p in self.points():
            if p in self.selectedPoints:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush(100, 150, 200, 200))


# ------------------------------
# Custom Edge Item with Hover and Click Interaction
# ------------------------------
class EdgeItem(pg.PlotCurveItem):
    """
    Custom PlotCurveItem representing an edge.
    It supports hover events (to display a tooltip with basic metadata)
    and emits a signal when clicked.
    """
    edgeClicked = pyqtSignal(dict)  # Emits edge metadata when clicked.

    def __init__(self, x, y, metadata, *args, **kwargs):
        super().__init__(x=x, y=y, *args, **kwargs)
        self.metadata = metadata
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        tooltip_text = (f"Protocol: {self.metadata.get('protocol', 'Unknown')}\n"
                        f"Packets: {self.metadata.get('packets', 0)}")
        QToolTip.showText(event.screenPos().toPoint(), tooltip_text)
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
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        # Enable built-in mouse wheel zooming and dragging (pan).
        self.plot_item.vb.setMouseMode(pg.ViewBox.RectMode)

        # Use custom draggable scatter plot for nodes.
        self.node_scatter = DraggableScatterPlotItem(
            size=30,
            pen=pg.mkPen(width=2, color='k'),
            brush=pg.mkBrush(100, 150, 200, 200)
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
        """
        self.clear_visualization()
        spots = []
        self.node_data = {}
        for node, position in self.node_positions.items():
            self.node_data[node] = self.graph.nodes[node]
            spots.append({
                'pos': position,
                'size': 30,
                'pen': {'color': 'k', 'width': 2},
                'brush': pg.mkBrush(100, 150, 200, 200),
                'symbol': 'o',
                'data': {'id': node, 'metadata': self.graph.nodes[node]}
            })
        self.node_scatter.setData(spots)

        self.edge_items = []
        self.edge_node_pairs = []
        for u, v, data in self.graph.edges(data=True):
            if u in self.node_positions and v in self.node_positions:
                pos_u = self.node_positions[u]
                pos_v = self.node_positions[v]
                edge_line = EdgeItem(
                    x=[pos_u[0], pos_v[0]],
                    y=[pos_u[1], pos_v[1]],
                    metadata=data,
                    pen=pg.mkPen(color=(100, 100, 100), width=1.5)
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
        Updates edge positions to follow node movements.
        """
        for edge_item, (u, v) in zip(self.edge_items, self.edge_node_pairs):
            if u in self.node_positions and v in self.node_positions:
                pos_u = self.node_positions[u]
                pos_v = self.node_positions[v]
                edge_item.setData(x=[pos_u[0], pos_v[0]], y=[pos_u[1], pos_v[1]])

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
        """
        Sets up the main window with:
         - A left panel containing the reset button and graph canvas.
         - A right side panel for metadata display.
         - Wiring of interactive signals (node/edge selection) to update the metadata panel.
        """
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        left_layout = QVBoxLayout()

        # Instantiate graph visualization first.
        self.graph_view = GraphVisualization()
        self.graph_view.setBackground('w')

        # Create control bar with "Reset View" button.
        control_bar = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.graph_view.reset_view)
        control_bar.addWidget(reset_btn)
        control_bar.addStretch()
        left_layout.addLayout(control_bar)
        left_layout.addWidget(self.graph_view, 1)

        # Metadata panel on right.
        self.metadata_panel = QLabel("Node/Edge metadata will appear here.")
        self.metadata_panel.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.metadata_panel.setWordWrap(True)
        self.metadata_panel.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        main_layout.addLayout(left_layout, 3)
        main_layout.addWidget(self.metadata_panel, 1)
        self.setCentralWidget(main_widget)

        # Wire signals to update metadata panel.
        self.graph_view.nodeSelected.connect(self.update_metadata_panel_for_node)
        self.graph_view.edgeSelected.connect(self.update_metadata_panel_for_edge)

    def _create_sample_graph(self):
        """
        Generates and displays a sample graph.
        """
        sample_graph = self.graph_view.create_sample_graph()
        self.graph_view.set_graph(sample_graph)
        self.statusBar().showMessage("Displaying sample graph - Drop a JSON file to load real data")

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
        text_lines = []
        for node, meta in node_metadata.items():
            text_lines.append(f"Node: {node}")
            for key, value in meta.items():
                text_lines.append(f"  {key}: {value}")
            text_lines.append("")
        self.metadata_panel.setText("\n".join(text_lines))

    def update_metadata_panel_for_edge(self, edge_metadata):
        """
        Updates the side panel with metadata for the selected edge.
        """
        if not edge_metadata:
            self.metadata_panel.setText("No edge selected.")
            return
        text_lines = ["Edge Metadata:"]
        for key, value in edge_metadata.items():
            text_lines.append(f"{key}: {value}")
        self.metadata_panel.setText("\n".join(text_lines))


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
                   duration=duration, src_port=src_port, dst_port=dst_port,
                   packet_details=packets)
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
