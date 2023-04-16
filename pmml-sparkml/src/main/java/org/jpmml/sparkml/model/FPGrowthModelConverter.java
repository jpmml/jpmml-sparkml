/*
 * Copyright (c) 2021 Villu Ruusmann
 *
 * This file is part of JPMML-SparkML
 *
 * JPMML-SparkML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * JPMML-SparkML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with JPMML-SparkML.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jpmml.sparkml.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.spark.ml.fpm.FPGrowthModel;
import org.apache.spark.sql.Row;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.OpType;
import org.dmg.pmml.association.AssociationModel;
import org.dmg.pmml.association.AssociationRule;
import org.dmg.pmml.association.Item;
import org.dmg.pmml.association.ItemRef;
import org.dmg.pmml.association.Itemset;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.AssociationRulesModelConverter;
import org.jpmml.sparkml.ItemSetFeature;
import org.jpmml.sparkml.SparkMLEncoder;
import scala.collection.JavaConversions;
import scala.collection.Seq;

public class FPGrowthModelConverter extends AssociationRulesModelConverter<FPGrowthModel> {

	public FPGrowthModelConverter(FPGrowthModel model){
		super(model);
	}

	@Override
	public List<Feature> getFeatures(SparkMLEncoder encoder){
		FPGrowthModel model = getTransformer();

		String itemsCol = model.getItemsCol();

		// Convert from plural to singular
		if(itemsCol.endsWith("s")){
			itemsCol = itemsCol.substring(0, itemsCol.length() - 1);
		}

		DataField transactionDataField = encoder.createDataField("transaction", OpType.CATEGORICAL, DataType.STRING);

		DataField itemDataField = encoder.createDataField(itemsCol, OpType.CATEGORICAL, DataType.STRING);

		Feature feature = new ItemSetFeature(encoder, itemDataField);

		return Collections.singletonList(feature);
	}

	@Override
	public AssociationModel encodeModel(Schema schema){
		FPGrowthModel model = getTransformer();

		List<? extends Feature> features = schema.getFeatures();

		SchemaUtil.checkSize(1, features);

		Feature feature = features.get(0);

		Map<String, Item> items = new LinkedHashMap<>();
		Map<List<String>, Itemset> itemsets = new LinkedHashMap<>();

		List<AssociationRule> associationRules = new ArrayList<>();

		List<Row> associationRuleRows = (model.associationRules()).collectAsList();
		for(Row associationRuleRow : associationRuleRows){
			List<String> antecedent = formatValues(JavaConversions.seqAsJavaList((Seq<?>)associationRuleRow.apply(0)));
			List<String> consequent = formatValues(JavaConversions.seqAsJavaList((Seq<?>)associationRuleRow.apply(1)));

			Double confidence = (Double)associationRuleRow.apply(2);

			// XXX
			Double lift = 0d;
			Double support = 0d;

			Itemset antecedentItemset = ensureItemset(feature, antecedent, itemsets, items);
			Itemset consequentItemset = ensureItemset(feature, consequent, itemsets, items);

			AssociationRule associationRule = new AssociationRule()
				.setAntecedent(antecedentItemset.requireId())
				.setConsequent(consequentItemset.requireId());

			associationRule = associationRule
				.setConfidence(confidence)
				.setLift(lift)
				.setSupport(support);

			associationRules.add(associationRule);
		}

		// XXX
		int numberOfTransactions = 0;

		MiningField transactionMiningField = ModelUtil.createMiningField("transaction", MiningField.UsageType.GROUP);

		MiningSchema miningSchema = new MiningSchema()
			.addMiningFields(transactionMiningField);

		AssociationModel associationModel = new AssociationModel(MiningFunction.ASSOCIATION_RULES, numberOfTransactions, model.getMinSupport(), model.getMinConfidence(), items.size(), itemsets.size(), associationRules.size(), miningSchema);

		(associationModel.getItems()).addAll(items.values());
		(associationModel.getItemsets()).addAll(itemsets.values());
		(associationModel.getAssociationRules()).addAll(associationRules);

		return associationModel;
	}

	static
	private Itemset ensureItemset(Feature feature, List<String> values, Map<List<String>, Itemset> itemsets, Map<String, Item> items){
		Itemset itemset = itemsets.get(values);

		if(itemset == null){
			itemset = new Itemset(String.valueOf(itemsets.size() + 1));

			for(String value : values){
				Item item = items.get(value);

				if(item == null){
					item = new Item(String.valueOf(items.size() + 1), value)
						// XXX: See SparkMLEncoder#encodePMML(Model)
						.setField(feature.getName());

					items.put(value, item);
				}

				itemset.addItemRefs(new ItemRef(item.getId()));
			}

			List<ItemRef> itemRefs = itemset.getItemRefs();
			if(itemRefs.size() > 1){
				Comparator<ItemRef> comparator = new Comparator<ItemRef>(){

					@Override
					public int compare(ItemRef left, ItemRef right){
						int leftId = Integer.parseInt(left.requireItemRef());
						int rightId = Integer.parseInt(right.requireItemRef());

						return Integer.compare(leftId, rightId);
					}
				};

				Collections.sort(itemRefs, comparator);
			}

			itemsets.put(values, itemset);
		}

		return itemset;
	}

	static
	public List<String> formatValues(List<?> values){
		return values.stream()
			.map(value -> ValueUtil.asString(value))
			.collect(Collectors.toList());
	}
}