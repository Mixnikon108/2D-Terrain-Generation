from typing import Set, Any, Optional, Callable, Dict, Tuple, List, Union
import numpy as np
import random
from typing import Tuple
from collections import deque
import pygame

#%%
class Node:
    """
    Representa un nodo en una estructura de datos de red, con conexiones a otros nodos.
    
    Attributes:
        x (Any): Valor horizontal o atributo x del nodo.
        y (Any): Valor vertical o atributo y del nodo.
        connections (Set['Node']): Conjunto de nodos conectados a este nodo.
        unique_id (int): Identificador único autoincremental para cada instancia de Node.
    """
    _current_id = 0  # Variable de clase para rastrear el ID actual

    def __init__(self, x: Any, y: Any) -> None:
        """
        Inicializa un nuevo nodo con valores específicos y un identificador único.
        
        Args:
            x (Any): Valor horizontal o atributo x del nodo.
            y (Any): Valor vertical o atributo y del nodo.
        """
        self.coordinates = np.array([x, y])
        self.connections: Set['Node'] = set()
        self.unique_id = self._get_unique_id()

    @classmethod
    def _get_unique_id(cls, reset: bool = False) -> int:
        """
        Obtiene un identificador único para la instancia, opcionalmente reseteando el contador.
        
        Args:
            reset (bool): Si es True, resetea el contador de ID a cero.
        
        Returns:
            int: El siguiente ID único para una nueva instancia de Node.
        """
        if reset:
            cls._current_id = 0
        else:
            cls._current_id += 1
        return cls._current_id

    def add_connection(self, other_node: 'Node') -> None:
        """
        Establece una conexión bidireccional entre este nodo y otro, asegurando que no se añadan duplicados.
        
        Args:
            other_node (Node): El nodo con el que se establecerá la conexión.
        
        Raises:
            AssertionError: Si other_node es el mismo que self.
        """
        assert other_node is not self, "A node cannot connect to itself."
        if other_node not in self.connections:
            self.connections.add(other_node)
            other_node.connections.add(self)

    def has_connection(self, other_node: 'Node') -> bool:
        """
        Verifica si existe una conexión con otro nodo.
        
        Args:
            other_node (Node): El nodo para verificar si existe una conexión.
        
        Returns:
            bool: True si existe la conexión, False de lo contrario.
        """
        return other_node in self.connections
    
    def remove_connection(self, other_node: 'Node') -> None:
        """
        Elimina la conexión bidireccional con otro nodo, si existe.
        
        Args:
            other_node (Node): El nodo del cual se eliminará la conexión.
        
        Raises:
            ValueError: Si el nodo especificado no está conectado.
        """
        if other_node in self.connections:
            self.connections.remove(other_node)
            other_node.connections.remove(self)
        else:
            raise ValueError("No connection exists with the specified node.")
    
    def __str__(self) -> str:
        """
        Proporciona una representación en forma de string del nodo, mostrando ID, coordenadas y conexiones.
        
        Returns:
            str: La representación en string del nodo.
        """
        connection_ids = ', '.join(str(node.unique_id) for node in self.connections)
        return (f"Node(ID: {self.unique_id}, Coordinates: {self.coordinates}, "
                f"Connections: [{connection_ids}])")

#%%
def cache_nodes(verbose: bool = False) -> Callable:
    """
    A decorator that caches the results of a function based on its node arguments.
    It assumes that each node argument has a unique attribute `unique_id`.

    Args:
        verbose (bool): If set to True, the decorator will print whether a cache hit
                        or miss occurred.

    Returns:
        Callable: A wrapped function that uses caching to optimize calls.
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[Tuple, Any] = {}
        
        def wrapper(*args, **kwargs) -> Any:
            # Extract unique identifiers from node arguments.
            nodes = [node.unique_id for node in [*args, *kwargs.values()]]
            key = tuple(sorted(nodes))
            
            # Optionally print the cache key.
            if verbose:
                print(f"Cache key: {key}")
            
            # Check if result is cached and return it, otherwise compute and cache.
            if key in cache:
                if verbose:
                    print('Cache hit.')
                return cache[key]
            else:
                if verbose:
                    print('Cache miss.')
                result = func(*args, **kwargs)
                cache[key] = result
                return result
        
        return wrapper
    
    return decorator

#%%
def random_perpendicular_point(start: np.ndarray, end: np.ndarray, roughtness: float = 0.3) -> np.ndarray:
    """
    Calculates a random point along the perpendicular bisector of the line segment defined by 'start' and 'end',
    scaled by a 'scale' factor.
    
    Args:
        start (np.ndarray): The starting point of the line segment in 2D space.
        end (np.ndarray): The ending point of the line segment in 2D space.
        roughtness (float): A factor that scales the random displacement along the perpendicular.
    
    Returns:
        np.ndarray: The computed 2D point along the perpendicular bisector.
    
    Raises:
        ValueError: If 'start' or 'end' are not 2-dimensional vectors.
        ZeroDivisionError: If 'start' and 'end' are identical, resulting in a zero-length segment.
    """
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("Both 'start' and 'end' must be 2-dimensional vectors.")

    segment_vector = end - start
    norm = np.linalg.norm(segment_vector)
    try:
        unit_segment_vector = segment_vector / norm
    except ZeroDivisionError:
        raise ZeroDivisionError("Cannot create a normal from a zero-length segment ('start' and 'end' are identical).")

    normal_vector = np.array([-unit_segment_vector[1], unit_segment_vector[0]])  # Rotate 90 degrees to get the normal
    
    alpha = (norm // 2.3) * roughtness
    displacement = random.uniform(-alpha, alpha)
    
    
    midpoint = (start + end) / 2
    
    return normal_vector * displacement + midpoint

#%%
@cache_nodes(verbose=False)
def create_new_node(node_a: Node, node_b: Node) -> Node:
    """
    Creates a new node at a random position perpendicular to the line between two connected nodes,
    modifies the connections to integrate the new node between them.

    Args:
        node_a (Node): The first node, connected to node_b.
        node_b (Node): The second node, connected to node_a.

    Returns:
        Node: The newly created node connected to node_a and node_b.

    Raises:
        AssertionError: If node_a and node_b are not connected.
        ValueError: For geometric calculation errors or invalid node operations.
    """
    if not node_a.has_connection(node_b):
        raise ValueError("Nodes must be connected to create a new node between them.")

    try:
        x, y = random_perpendicular_point(node_a.coordinates, node_b.coordinates)
        new_node = Node(x, y)
        new_node.add_connection(node_a)
        new_node.add_connection(node_b)
        node_a.remove_connection(node_b)
        return new_node
    except Exception as e:
        raise ValueError(f"An error occurred while creating a new node: {e}")



#%%
class FIFO:
    """
    Representa una cola FIFO (First In, First Out) que maneja elementos en un orden de llegada.
    
    Esta estructura de datos permite agregar elementos al final y retirar el primer elemento añadido,
    operando bajo el principio de "el primero en entrar es el primero en salir".
    """

    def __init__(self) -> None:
        """Inicializa una cola FIFO vacía."""
        self.items = deque()

    def enqueue(self, item: Union[Any, List[Any]], is_list: bool = False) -> None:
        """
        Añade un elemento o una lista de elementos al final de la cola.
        
        Si el elemento es una lista y `is_list` es True, cada elemento de la lista
        se añade individualmente a la cola. De lo contrario, el elemento completo se añade como uno solo.
        
        Args:
            item (Union[Any, List[Any]]): El elemento o lista de elementos a ser añadidos a la cola.
            is_list (bool): Indica si `item` es una lista de elementos.
        """
        if is_list and isinstance(item, list):
            self.items.extend(item)  # Añade cada elemento de la lista a la cola
        else:
            self.items.append(item)  # Añade el elemento único a la cola

    def dequeue(self) -> Any:
        """
        Elimina y retorna el elemento al frente de la cola.
        
        Returns:
            Any: El primer elemento de la cola.
        
        Raises:
            IndexError: Si se intenta retirar un elemento de una cola vacía.
        """
        if self.is_empty():
            raise IndexError("dequeue from an empty queue")
        return self.items.popleft()

    def is_empty(self) -> bool:
        """
        Verifica si la cola está vacía.
        
        Returns:
            bool: True si la cola no tiene elementos, False de lo contrario.
        """
        return len(self.items) == 0

    def size(self) -> int:
        """
        Retorna el número de elementos en la cola.
        
        Returns:
            int: La cantidad de elementos en la cola.
        """
        return len(self.items)

    def __str__(self) -> str:
        """
        Proporciona una representación en forma de string de la cola.
        
        Returns:
            str: La representación de la cola con todos sus elementos.
        """
        return f"FIFO({list(self.items)})"
  
#%%
class Triangle:
    def __init__(self, node_a: Node, node_b: Node, node_c: Node): 
        """
        Initializes a Triangle with three nodes.
        
        Args:
            node_a (Node): The first vertex of the triangle.
            node_b (Node): The second vertex of the triangle.
            node_c (Node): The third vertex of the triangle.
        """
        self.node_a = node_a
        self.node_b = node_b
        self.node_c = node_c
    
        self.create_edges()
    
    def create_edges(self) -> None:
        """
        Establishes bidirectional connections between the vertices of the triangle.
        """
        self.node_a.add_connection(self.node_b)
        self.node_a.add_connection(self.node_c)
        self.node_b.add_connection(self.node_c)
        
    def subdivide(self) -> Tuple['Triangle', 'Triangle', 'Triangle', 'Triangle']:
        """
        Subdivides the triangle into four smaller triangles by creating new nodes
        at the midpoints of each edge.
        
        Returns:
            Tuple[Triangle, Triangle, Triangle, Triangle]: A tuple containing the four new triangles.
        """
        node_d = create_new_node(self.node_a, self.node_b)
        node_e = create_new_node(self.node_b, self.node_c)
        node_f = create_new_node(self.node_a, self.node_c)
        
        triangle_1 = Triangle(node_d, self.node_b, node_e)
        triangle_2 = Triangle(self.node_a, node_d, node_f)
        triangle_3 = Triangle(node_f, node_e, self.node_c)
        triangle_4 = Triangle(node_d, node_e, node_f)
        
        return [triangle_1, triangle_2, triangle_3, triangle_4]
    
    def draw(self, screen: Any, color: Tuple[int, int, int]) -> None:
        """
        Draws the triangle on a Pygame screen using the provided color.
        
        Args:
            screen (Any): The Pygame screen where the triangle will be drawn.
            color (Tuple[int, int, int]): The RGB color tuple for the triangle.
        """
        pygame.draw.polygon(screen, color, [
            self.node_a.coordinates, 
            self.node_b.coordinates, 
            self.node_c.coordinates
        ], 1)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Triangle.
        """
        return f'Vertex 1: {self.node_a}, Vertex 2: {self.node_b}, Vertex 3: {self.node_c}'
