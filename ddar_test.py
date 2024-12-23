"""Unit tests for ddar.py."""
import unittest

from absl.testing import absltest
import ddar
import graph as gh
import problem as pr


class DDARTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
    cls.rules = pr.Theorem.from_txt_file('rules.txt', to_dict=True)

  def test_orthocenter_should_fail(self):
    txt = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b ? perp a d b c'  # pylint: disable=line-too-long
    p = pr.Problem.from_txt(txt)
    g, _ = gh.Graph.build_problem(p, DDARTest.defs)

    ddar.solve(g, DDARTest.rules, p, max_level=1000)
    goal_args = g.names2nodes(p.goal.args)
    self.assertFalse(g.check(p.goal.name, goal_args))

  def test_orthocenter_aux_should_succeed(self):
    txt = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b; e = on_line e a c, on_line e b d ? perp a d b c'  # pylint: disable=line-too-long
    p = pr.Problem.from_txt(txt)
    g, _ = gh.Graph.build_problem(p, DDARTest.defs)

    ddar.solve(g, DDARTest.rules, p, max_level=1000)
    goal_args = g.names2nodes(p.goal.args)
    self.assertTrue(g.check(p.goal.name, goal_args))

  def test_incenter_excenter_should_succeed(self):
    # Note that this same problem should fail in dd_test.py
    p = pr.Problem.from_txt(
        'a b c = triangle a b c; d = incenter d a b c; e = excenter e a b c ?'
        ' perp d c c e'
    )  # pylint: disable=line-too-long
    g, _ = gh.Graph.build_problem(p, DDARTest.defs)

    ddar.solve(g, DDARTest.rules, p, max_level=1000)
    goal_args = g.names2nodes(p.goal.args)
    self.assertTrue(g.check(p.goal.name, goal_args))


if __name__ == '__main__':
  absltest.main()
