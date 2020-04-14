import numpy as np

def normalize(v):
  """Return vector in same direction but with length 1."""
  return np.array(v) / np.linalg.norm(v)

def rad_to_deg(rad):
  return 180 * rad / np.pi

def angle_between(v1, v2):
  """Return angle (in radians) between v1 and v2"""
  n_v1 = normalize(v1)
  n_v2 = normalize(v2)

  # s = 1 means the angle is positive, s=-1 means negative angle
  cross_product = np.linalg.det([n_v1, n_v2])
  s = np.sign(cross_product)
  if s == 0:
    s = 1

  # dot_product = cos(angle between n_v1 and n_v2)
  dot_product = np.dot(n_v1, n_v2)

  # due to numerical fuzziness, it's possible for
  # dot_product to go slightly out of [-1, 1],
  # so ensure it's within these bounds
  dot_product = np.min([dot_product, 1])
  dot_product = np.max([dot_product, -1])

  return s * np.arccos(dot_product)

def compute_total_angle(points):
  """Return the total "directed" outside angle of the polygon defined by the
  given set of points.
  """
  assert len(points) > 2, 'Need at least 3 points for a complete polygon.'

  if np.isclose(points[0][0], points[-1][0], rtol=1e-8) and np.isclose(points[0][1], points[-1][1], rtol=1e-8):
    # last point is same as first point; trim off the last point for the computation below
    points = points[:-1]

  cumulative_angle = 0
  for i in range(len(points)):
    j = (i + 1) % len(points)
    k = (i + 2) % len(points)
    v1 = np.array(points[j]) - np.array(points[i])
    v2 = np.array(points[k]) - np.array(points[j])
    cumulative_angle += angle_between(v1, v2)
  return cumulative_angle

def compute_n_rotations(points):
  """Compute how many times the given polygon "winds around", and in which direction."""
  total_angle = rad_to_deg(compute_total_angle(points))
  n_rotations = total_angle / 360.0
  round_n_rotations = np.round(n_rotations)
  # expect n_rotations to be very close to an integer
  assert np.isclose(np.round(n_rotations, 3), n_rotations, atol=1e-8)
  return round_n_rotations

########## A few tests ############
def check(expected, actual, rtol=1e-07):
  assert(np.isclose(expected, actual, rtol))

def test_angle_between():
  check(np.pi / 2, angle_between([10,0], [0,3]))
  check(-np.pi / 2, angle_between([0,10], [100,0]))
  check(5 * np.pi / 6, angle_between([10,0], [-np.sqrt(3), 1]))
  check(np.pi / 6, angle_between([10,0], [np.sqrt(3), 1]))
  check(-5 * np.pi / 6, angle_between([10,0], [-np.sqrt(3), -1]))
  check(-np.pi / 4, angle_between([0,1], [5,5]))

def test_compute_total_angle():
  # a triangle - counterclockwise
  points = [[0,0], [1,0], [1,1]]
  check(2 * np.pi, compute_total_angle(points))
  # square - counterclockwise
  points = [[0,0], [1,0], [1,1], [0, 1], [0,0]]
  check(2 * np.pi, compute_total_angle(points))
  # non-convex polygon - counterclockwise
  points = [[0,0], [1,0], [1,-1], [2,0], [2, 10]]
  check(2 * np.pi, compute_total_angle(points))

  # a triangle - *clockwise*
  points = [[0,0], [1,1], [1,0], [0,0]]
  check(-2 * np.pi, compute_total_angle(points))
  # square - *clockwise*
  points = [[0,0], [1,0], [1,1], [0, 1]]
  points.reverse()
  check(-2 * np.pi, compute_total_angle(points))
  # non-convex polygon - *clockwise*
  points = [[0,0], [1,0], [1,-1], [2,0], [2, 10]]
  points.reverse()
  check(-2 * np.pi, compute_total_angle(points))

if __name__ == "__main__":
  test_angle_between()
  test_compute_total_angle()
  print("All tests passed!")

############# A few examples ###################################

##### example for how to get the orientation of a polygon:
##### this snippet computes the orientation for each polygon in
##### your dataset
# orientation_list = []
# for i in range(len(nymap['features'])):
#   orientation_of_polygon = compute_n_rotations(nymap['features'][i]['geometry']['coordinates'][0])
#   orientation_list.append(orientation_of_polygon)

##### example for how to change the orientation of a set of points:
##### the reverse() function reverses the order of the points
# for i in range(len(nymap['features'])):
#   nymap['features'][i]['geometry']['coordinates'][0].reverse()
