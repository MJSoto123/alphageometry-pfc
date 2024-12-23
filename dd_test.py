"""Unit tests for dd."""
import unittest

from absl.testing import absltest
import dd
import graph as gh
import problem as pr


MAX_LEVEL = 1000


class DDTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
    cls.rules = pr.Theorem.from_txt_file('rules.txt', to_dict=True)

  def test_imo_2022_p4_should_succeed(self):
    p = pr.Problem.from_txt(
        'a b = segment a b; g1 = on_tline g1 a a b; g2 = on_tline g2 b b a; m ='
        ' on_circle m g1 a, on_circle m g2 b; n = on_circle n g1 a, on_circle n'
        ' g2 b; c = on_pline c m a b, on_circle c g1 a; d = on_pline d m a b,'
        ' on_circle d g2 b; e = on_line e a c, on_line e b d; p = on_line p a'
        ' n, on_line p c d; q = on_line q b n, on_line q c d ? cong e p e q'
    )
    g, _ = gh.Graph.build_problem(p, DDTest.defs)
    goal_args = g.names2nodes(p.goal.args)

    success = False
    for level in range(MAX_LEVEL):
      added, _, _, _ = dd.bfs_one_level(g, DDTest.rules, level, p)
      if g.check(p.goal.name, goal_args):
        success = True
        break
      if not added:  # saturated
        break

    self.assertTrue(success)

  def test_incenter_excenter_should_fail(self):
    p = pr.Problem.from_txt(
        'a b c = triangle a b c; d = incenter d a b c; e = excenter e a b c ?'
        ' perp d c c e'
    )
    g, _ = gh.Graph.build_problem(p, DDTest.defs)
    goal_args = g.names2nodes(p.goal.args)

    success = False
    for level in range(MAX_LEVEL):
      added, _, _, _ = dd.bfs_one_level(g, DDTest.rules, level, p)
      if g.check(p.goal.name, goal_args):
        success = True
        break
      if not added:  # saturated
        break

    self.assertFalse(success)


if __name__ == '__main__':
  absltest.main()
