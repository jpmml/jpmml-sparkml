package org.jpmml.sparkml;

import org.dmg.pmml.*;

import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sparkml.model.TreeModelUtil;

import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class VisitorsTest {
    TreeModel treeModel;
    static int numberOfNodes = 7;

    //Build some tree with $numberOfNodes Nodes

    @Before
    public void buildTreeNodes() {
        List<Node> nodes = new ArrayList<>();

        for (int i = 0; i < numberOfNodes; ++i) nodes.add(new Node());

        nodes.get(0).addNodes(nodes.get(1), nodes.get(2), nodes.get(3));

        nodes.get(2).addNodes(nodes.get(4));
        nodes.get(3).addNodes(nodes.get(5));

        nodes.get(4).addNodes(nodes.get(6));

        treeModel = new TreeModel(MiningFunctionType.CLASSIFICATION, new MiningSchema(), nodes.get(0));
    }

    class IndexCollector extends AbstractVisitor {
        Set<Integer> indices = new HashSet<>();

        @Override
        public VisitorAction visit(Node node) {
            Integer ind = Integer.parseInt(node.getId());
            indices.add(ind);

            return super.visit(node);
        }

        public Set<Integer> getIndices() {
            return indices;
        }
    }

    @Test
    public void indexNodesTest() {

        IndexCollector indexCollector = new IndexCollector();

        TreeModelUtil.indexNodes(treeModel);

        indexCollector.applyTo(treeModel);

        Set<Integer> actual = indexCollector.getIndices();
        Set<Integer> expected = new HashSet(Arrays.asList(1, 2, 3, 4, 5, 6, 7));

        assertEquals(true, actual.equals(expected));
    }
}
